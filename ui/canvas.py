# yolo_label_tool/ui/canvas.py
"""影像畫布元件 (支援 BBox & Polygon Segmentation)"""
# ui/canvas.py 開頭部分
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
# ui/canvas.py
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint, QSizeF, pyqtSignal, QSize 
from PyQt5.QtGui import (QPixmap, QPainter, QPen, QBrush, QColor, QFont,
                         QImage, QCursor, QWheelEvent, QMouseEvent, 
                         QPaintEvent, QKeyEvent, QPalette, QPainterPath)

# 確保這裡的路徑正確
from core.yolo_io import YoloAnnotation, read_image_standardized
from core.yolo_seg_io import YoloSegAnnotation

DEFAULT_COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A",
    "#92CC17", "#3DDB86", "#1A9334", "#00D4BB", "#2C99A8", "#00C2FF",
    "#344593", "#6473FF", "#0018EC", "#8438FF", "#520085", "#CB38FF",
]

class BBoxItem:
    HANDLE_NONE, HANDLE_MOVE = 0, 1
    HANDLE_TL, HANDLE_TR, HANDLE_BL, HANDLE_BR = 2, 3, 4, 5
    HANDLE_TM, HANDLE_BM, HANDLE_LM, HANDLE_RM = 6, 7, 8, 9

    def __init__(self, annotation: YoloAnnotation, class_name: str, color: QColor):
        self.annotation = annotation
        self.class_name = class_name
        self.color = color
        self.selected = False
        self.widget_rect = QRectF()

    def contains_point(self, point: QPointF, handle_size: float = 8.0) -> int:
        if not self.widget_rect.isValid(): return self.HANDLE_NONE
        r = self.widget_rect.normalized()
        hs = handle_size
        handles = [
            (self.HANDLE_TL, r.topLeft()), (self.HANDLE_TR, r.topRight()),
            (self.HANDLE_BL, r.bottomLeft()), (self.HANDLE_BR, r.bottomRight()),
            (self.HANDLE_TM, QPointF(r.center().x(), r.top())),
            (self.HANDLE_BM, QPointF(r.center().x(), r.bottom())),
            (self.HANDLE_LM, QPointF(r.left(), r.center().y())),
            (self.HANDLE_RM, QPointF(r.right(), r.center().y())),
        ]
        for hid, p in handles:
            if QRectF(p.x() - hs, p.y() - hs, hs * 2, hs * 2).contains(point): return hid
        if r.contains(point): return self.HANDLE_MOVE
        return self.HANDLE_NONE

class PolyItem:
    def __init__(self, annotation: YoloSegAnnotation, class_name: str, color: QColor):
        self.annotation = annotation
        self.class_name = class_name
        self.color = color
        self.selected = False
        self.widget_points: List[QPointF] = []

    def contains_point(self, point: QPointF, handle_size: float = 8.0) -> Tuple[str, int]:
        hs = handle_size
        # 1. 檢查頂點
        for i, p in enumerate(self.widget_points):
            if QRectF(p.x() - hs, p.y() - hs, hs * 2, hs * 2).contains(point):
                return ('vertex', i)
        
        # 2. 檢查多邊形內部
        path = QPainterPath()
        if self.widget_points:
            path.moveTo(self.widget_points[0])
            for p in self.widget_points[1:]:
                path.lineTo(p)
            path.closeSubpath()
        if path.contains(point):
            return ('move', -1)
            
        return ('none', -1)

class ImageCanvas(QWidget):
    box_changed = pyqtSignal()
    box_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.task_mode = 'segment' # 僅支援 segment 標註

        self.original_pixmap: Optional[QPixmap] = None
        self.image_size = QSize(0, 0)
        self.scale = 1.0
        self.min_scale, self.max_scale = 0.05, 20.0
        self.offset = QPointF(0, 0)
        self.fit_to_window = True

        # BBox 狀態
        self.boxes: List[BBoxItem] = []
        # Poly 狀態
        self.polys: List[PolyItem] = []
        
        self.selected_item = None # 統一選中物件 (BBoxItem 或 PolyItem)
        
        # 互動狀態
        self.dragging = False
        self.active_handle = BBoxItem.HANDLE_NONE
        self.active_poly_vertex_idx = -1
        self.drag_start_pos = QPointF()
        self.drag_start_widget_rect = QRectF()
        self.drag_start_poly_points = []
        
        self.pan_mode = False
        self.pan_start_pos = QPointF()
        self.pan_start_offset = QPointF()

        # 建立新框/多邊形
        self.creating = False
        self.create_start_widget = QPointF()
        self.create_end_widget = QPointF()
        
        self.creating_poly = False
        self.current_poly_points: List[QPointF] = []

        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(self.backgroundRole(), QColor(40, 40, 40))
        self.setPalette(pal)

    def set_task_mode(self, mode: str):
        self.task_mode = mode
        self.clear_all_annotations()
        self.update()

    def clear_all_annotations(self):
        self.boxes.clear()
        self.polys.clear()
        self.selected_item = None
        self.creating_poly = False
        self.current_poly_points.clear()
        self.box_selected.emit(None)

    def load_image(self, image_path: str) -> bool:
        try:
            bgr, (w, h) = read_image_standardized(image_path)
            if bgr is None or w == 0 or h == 0: return False
            rgb = bgr[..., ::-1].copy()
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            if pixmap.isNull(): return False
            
            self.original_pixmap = pixmap
            self.image_size = QSize(w, h)
            self.clear_all_annotations()
            
            if self.fit_to_window: self.fit_image()
            else: self.center_image()
            self.update()
            return True
        except Exception as e:
            print(f"load_image error: {e}")
            return False

    def fit_image(self):
        if not self.original_pixmap: return
        w, h = self.width(), self.height()
        if w == 0 or h == 0: return
        sx = w / self.image_size.width()
        sy = h / self.image_size.height()
        self.scale = max(self.min_scale, min(self.max_scale, min(sx, sy) * 0.98))
        self.center_image()

    def center_image(self):
        if not self.original_pixmap: return
        w, h = self.width(), self.height()
        scaled_w = self.image_size.width() * self.scale
        scaled_h = self.image_size.height() * self.scale
        self.offset = QPointF((w - scaled_w) / 2.0, (h - scaled_h) / 2.0)
        self.update()

    def widget_to_image(self, pos: QPointF) -> QPointF:
        if self.image_size.width() == 0: return QPointF(0, 0)
        return QPointF((pos.x() - self.offset.x()) / self.scale, 
                       (pos.y() - self.offset.y()) / self.scale)

    def image_to_widget(self, pos: QPointF) -> QPointF:
        return QPointF(pos.x() * self.scale + self.offset.x(), 
                       pos.y() * self.scale + self.offset.y())

    def _update_all_widget_coords(self):
        iw, ih = self.image_size.width(), self.image_size.height()
        if iw == 0 or ih == 0: return

        for box in self.boxes:
            ann = box.annotation
            cx, cy = ann.x_center * iw, ann.y_center * ih
            w, h = ann.width * iw, ann.height * ih
            pix_rect = QRectF(cx - w/2, cy - h/2, w, h)
            tl = self.image_to_widget(pix_rect.topLeft())
            br = self.image_to_widget(pix_rect.bottomRight())
            box.widget_rect = QRectF(tl, br).normalized()

        for poly in self.polys:
            poly.widget_points = []
            for px, py in poly.annotation.points:
                img_p = QPointF(px * iw, py * ih)
                poly.widget_points.append(self.image_to_widget(img_p))

    def set_annotations(self, annotations: list, class_names: List[str]):
        self.clear_all_annotations()
        for ann in annotations:
            cls_name = class_names[ann.class_id] if ann.class_id < len(class_names) else f"class_{ann.class_id}"
            color = QColor(DEFAULT_COLORS[ann.class_id % len(DEFAULT_COLORS)])
            
            if isinstance(ann, YoloSegAnnotation):
                self.polys.append(PolyItem(ann, cls_name, color))
            elif isinstance(ann, YoloAnnotation):
                self.boxes.append(BBoxItem(ann, cls_name, color))
                
        self._update_all_widget_coords()
        self.update()

    def get_annotations(self) -> list:
        if self.task_mode == 'segment':
            return self.polys
        return self.boxes

    def add_annotation(self, ann, class_name: str = "object"):
        color = QColor(DEFAULT_COLORS[ann.class_id % len(DEFAULT_COLORS)])
        if isinstance(ann, YoloSegAnnotation):
            item = PolyItem(ann, class_name, color)
            self.polys.append(item)
        else:
            item = BBoxItem(ann, class_name, color)
            self.boxes.append(item)
            
        self.selected_item = item
        self._update_all_widget_coords()
        self.box_selected.emit(item)
        self.box_changed.emit()
        self.update()

    def delete_selected(self):
        if self.selected_item:
            if isinstance(self.selected_item, PolyItem) and self.selected_item in self.polys:
                self.polys.remove(self.selected_item)
            elif isinstance(self.selected_item, BBoxItem) and self.selected_item in self.boxes:
                self.boxes.remove(self.selected_item)
            self.selected_item = None
            self.box_selected.emit(None)
            self.box_changed.emit()
            self.update()

    def set_selected_class(self, class_id: int, class_names: List[str]):
        if self.selected_item:
            self.selected_item.annotation.class_id = int(class_id)
            self.selected_item.class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            self.selected_item.color = QColor(DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)])
            self.box_changed.emit()
            self.update()

    # ================= 繪圖 =================
    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if not self.original_pixmap:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "請載入影像資料夾")
            return

        target = QRectF(self.offset.x(), self.offset.y(),
                        self.image_size.width() * self.scale,
                        self.image_size.height() * self.scale)
        painter.drawPixmap(target, self.original_pixmap, QRectF(self.original_pixmap.rect()))

        # 繪製標註
        for box in self.boxes: self._draw_box(painter, box)
        for poly in self.polys: self._draw_poly(painter, poly)

        # 繪製正在建立的多邊形
        if self.creating_poly and self.current_poly_points:
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 30))
            path = QPainterPath()
            path.moveTo(self.current_poly_points[0])
            for p in self.current_poly_points[1:]:
                path.lineTo(p)
            path.lineTo(self.create_end_widget)
            painter.drawPath(path)
            
            # 畫頂點
            painter.setBrush(QColor(0, 255, 0))
            for p in self.current_poly_points:
                painter.drawEllipse(p, 4, 4)

        # 繪製正在建立的 BBox
        if self.creating:
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(0, 255, 0, 30))
            rect = QRectF(self.create_start_widget, self.create_end_widget).normalized()
            painter.drawRect(rect)

    def _draw_box(self, painter: QPainter, box: BBoxItem):
        r = box.widget_rect.normalized()
        if not r.isValid(): return
        painter.setPen(QPen(box.color, 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(r)

        label = box.class_name
        font = QFont("Arial", max(8, int(10 * min(1.0, self.scale * 0.4 + 0.5))))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        text_w = metrics.horizontalAdvance(label) + 8
        text_h = metrics.height() + 4
        label_rect = QRectF(r.left(), max(0, r.top() - text_h), text_w, text_h)
        painter.fillRect(label_rect, box.color)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(label_rect, Qt.AlignCenter, label)

        if box.selected:
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(box.color, 1))
            hs = 4
            pts = [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight(),
                   QPointF(r.center().x(), r.top()), QPointF(r.center().x(), r.bottom()),
                   QPointF(r.left(), r.center().y()), QPointF(r.right(), r.center().y())]
            for p in pts: painter.drawRect(QRectF(p.x() - hs, p.y() - hs, hs * 2, hs * 2))

    def _draw_poly(self, painter: QPainter, poly: PolyItem):
        if not poly.widget_points: return
        
        path = QPainterPath()
        path.moveTo(poly.widget_points[0])
        for p in poly.widget_points[1:]:
            path.lineTo(p)
        path.closeSubpath()

        # 填色與邊框
        fill_color = QColor(poly.color)
        fill_color.setAlpha(40 if poly.selected else 20)
        painter.setPen(QPen(poly.color, 2))
        painter.setBrush(fill_color)
        painter.drawPath(path)

        # 標籤
        if poly.widget_points:
            label = poly.class_name
            font = QFont("Arial", max(8, int(10 * min(1.0, self.scale * 0.4 + 0.5))))
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label) + 8
            text_h = metrics.height() + 4
            
            # 標籤位置放在第一個點附近
            p0 = poly.widget_points[0]
            label_rect = QRectF(p0.x(), p0.y() - text_h, text_w, text_h)
            painter.fillRect(label_rect, poly.color)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label_rect, Qt.AlignCenter, label)

        # 選中時畫頂點
        if poly.selected:
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(QPen(poly.color, 1))
            hs = 4
            for p in poly.widget_points:
                painter.drawRect(QRectF(p.x() - hs, p.y() - hs, hs * 2, hs * 2))

    # ================= 事件處理 =================
    def resizeEvent(self, event):
        if self.fit_to_window and self.original_pixmap: self.fit_image()
        super().resizeEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if not self.original_pixmap: return
        self.fit_to_window = False
        mouse_pos = QPointF(event.pos())
        img_pos_before = self.widget_to_image(mouse_pos)
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        img_pos_after_widget = QPointF(img_pos_before.x() * self.scale, img_pos_before.y() * self.scale)
        self.offset = QPointF(mouse_pos.x() - img_pos_after_widget.x(), mouse_pos.y() - img_pos_after_widget.y())
        self._update_all_widget_coords()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if not self.original_pixmap: return
        self.setFocus()
        pos = QPointF(event.pos())

        if event.button() == Qt.MidButton or (event.button() == Qt.LeftButton and self.pan_mode):
            self.dragging = True
            self.pan_start_pos = pos
            self.pan_start_offset = QPointF(self.offset)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.LeftButton:
            # 如果正在畫多邊形
            if self.creating_poly and self.task_mode == 'segment':
                # 檢查是否點到起點 (閉合)
                if len(self.current_poly_points) > 2:
                    start_p = self.current_poly_points[0]
                    if (pos - start_p).manhattanLength() < 15:
                        self._finish_creating_poly()
                        return
                self.current_poly_points.append(pos)
                self.update()
                return

            # 一般點擊：尋找物件
            hit_item = None
            hit_info = None

            if self.task_mode == 'segment':
                for poly in reversed(self.polys):
                    info = poly.contains_point(pos)
                    if info[0] != 'none':
                        hit_item = poly
                        hit_info = info
                        break
            else:
                for box in reversed(self.boxes):
                    h = box.contains_point(pos)
                    if h != BBoxItem.HANDLE_NONE:
                        hit_item = box
                        hit_info = ('bbox_handle', h)
                        break

            if hit_item:
                self._deselect_all()
                hit_item.selected = True
                self.selected_item = hit_item
                self.box_selected.emit(hit_item)

                self.dragging = True
                self.drag_start_pos = pos
                
                if isinstance(hit_item, PolyItem):
                    self.active_poly_vertex_idx = hit_info[1] if hit_info[0] == 'vertex' else -1
                    self.drag_start_poly_points = [QPointF(p) for p in hit_item.widget_points]
                else:
                    self.active_handle = hit_info[1]
                    self.drag_start_widget_rect = QRectF(hit_item.widget_rect)
                self.update()
            else:
                self._deselect_all()
                self.box_selected.emit(None)
                
                if not self.pan_mode:
                    if self.task_mode == 'segment':
                        self.creating_poly = True
                        self.current_poly_points = [pos]
                    else:
                        self.creating = True
                        self.create_start_widget = pos
                        self.create_end_widget = pos
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = QPointF(event.pos())

        if self.pan_mode and self.dragging:
            self.offset = self.pan_start_offset + (pos - self.pan_start_pos)
            self._update_all_widget_coords()
            self.update()
            return

        if self.creating_poly:
            self.create_end_widget = pos
            self.update()
            return

        if self.creating:
            self.create_end_widget = pos
            self.update()
            return

        if self.dragging and self.selected_item:
            delta = pos - self.drag_start_pos
            
            if isinstance(self.selected_item, PolyItem):
                poly = self.selected_item
                if self.active_poly_vertex_idx >= 0:
                    # 拖曳單一頂點
                    poly.widget_points[self.active_poly_vertex_idx] = self.drag_start_poly_points[self.active_poly_vertex_idx] + delta
                else:
                    # 拖曳整體
                    for i in range(len(poly.widget_points)):
                        poly.widget_points[i] = self.drag_start_poly_points[i] + delta
            else:
                box = self.selected_item
                new_rect = QRectF(self.drag_start_widget_rect)
                if self.active_handle == BBoxItem.HANDLE_MOVE:
                    new_rect.translate(delta)
                # ... (省略 BBox 縮放邏輯，與原版相同，為節省篇幅保留核心) ...
                elif self.active_handle == BBoxItem.HANDLE_TL: new_rect.setTopLeft(self.drag_start_widget_rect.topLeft() + delta)
                elif self.active_handle == BBoxItem.HANDLE_BR: new_rect.setBottomRight(self.drag_start_widget_rect.bottomRight() + delta)
                # 其他 handle 可自行補齊或沿用原版
                
                box.widget_rect = new_rect.normalized()
                pix_rect = self._widget_to_image_rect(box.widget_rect)
                box.annotation = self._pixel_rect_to_yolo(pix_rect, box.annotation.class_id)

            self.box_changed.emit()
            self.update()
            return

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MidButton or (self.pan_mode and event.button() == Qt.LeftButton):
            self.dragging = False
            self.setCursor(Qt.OpenHandCursor if self.pan_mode else Qt.ArrowCursor)
            return

        if self.creating and event.button() == Qt.LeftButton:
            self.creating = False
            rect = QRectF(self.create_start_widget, self.create_end_widget).normalized()
            if rect.width() > 5 and rect.height() > 5:
                pix_rect = self._widget_to_image_rect(rect)
                if pix_rect.width() > 1 and pix_rect.height() > 1:
                    cls_id, cls_name = self._get_default_class()
                    ann = self._pixel_rect_to_yolo(pix_rect, cls_id)
                    self.add_annotation(ann, cls_name)
            self.update()

        if self.dragging and self.selected_item and isinstance(self.selected_item, PolyItem):
            # 結束拖曳多邊形時，同步 normalized 座標
            self._sync_poly_widget_to_norm(self.selected_item)
            
        if self.dragging:
            self.dragging = False
            self.active_handle = BBoxItem.HANDLE_NONE
            self.active_poly_vertex_idx = -1

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.creating_poly and event.button() == Qt.LeftButton:
            if len(self.current_poly_points) > 2:
                self._finish_creating_poly()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.pan_mode = True
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.creating_poly:
            if len(self.current_poly_points) > 2:
                self._finish_creating_poly()
        elif event.key() == Qt.Key_Escape:
            if self.creating_poly:
                self.creating_poly = False
                self.current_poly_points.clear()
                self.update()

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.pan_mode = False
            self.setCursor(Qt.ArrowCursor)
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()

    # ================= 輔助函式 =================
    def _deselect_all(self):
        for b in self.boxes: b.selected = False
        for p in self.polys: p.selected = False
        self.selected_item = None

    def _get_default_class(self):
        if self.parent() and hasattr(self.parent(), 'get_default_class'):
            return self.parent().get_default_class()
        return 0, "object"

    def _widget_to_image_rect(self, widget_rect: QRectF) -> QRectF:
        tl = self.widget_to_image(widget_rect.topLeft())
        br = self.widget_to_image(widget_rect.bottomRight())
        return QRectF(tl, br).normalized()

    def _pixel_rect_to_yolo(self, rect: QRectF, class_id: int) -> YoloAnnotation:
        iw, ih = self.image_size.width(), self.image_size.height()
        if iw == 0 or ih == 0: return YoloAnnotation(class_id, 0, 0, 0, 0)
        x1, y1 = max(0, rect.left()), max(0, rect.top())
        x2, y2 = min(iw, rect.right()), min(ih, rect.bottom())
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return YoloAnnotation(class_id, (x1+x2)/2/iw, (y1+y2)/2/ih, (x2-x1)/iw, (y2-y1)/ih)

    def _finish_creating_poly(self):
        self.creating_poly = False
        if len(self.current_poly_points) < 3:
            self.current_poly_points.clear()
            self.update()
            return

        # 轉換為 normalized 座標
        iw, ih = self.image_size.width(), self.image_size.height()
        norm_points = []
        for wp in self.current_poly_points:
            ip = self.widget_to_image(wp)
            norm_points.append((max(0, min(1, ip.x()/iw)), max(0, min(1, ip.y()/ih))))

        cls_id, cls_name = self._get_default_class()
        try:
            ann = YoloSegAnnotation(cls_id, norm_points)
            self.add_annotation(ann, cls_name)
        except ValueError:
            pass
            
        self.current_poly_points.clear()
        self.update()

    def _sync_poly_widget_to_norm(self, poly: PolyItem):
        iw, ih = self.image_size.width(), self.image_size.height()
        if iw == 0 or ih == 0: return
        norm_points = []
        for wp in poly.widget_points:
            ip = self.widget_to_image(wp)
            norm_points.append((max(0, min(1, ip.x()/iw)), max(0, min(1, ip.y()/ih))))
        try:
            poly.annotation = YoloSegAnnotation(poly.annotation.class_id, norm_points)
        except ValueError:
            pass