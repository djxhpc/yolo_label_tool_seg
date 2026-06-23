# yolo_label_tool/ui/main_window.py
import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QListWidget, QListWidgetItem, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QGroupBox, QComboBox,
                             QSpinBox, QDoubleSpinBox, QSplitter, QStatusBar,
                             QAction, QToolBar, QApplication, QLineEdit,
                             QFormLayout, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QEvent
from PyQt5.QtGui import QKeySequence

from ui.canvas import ImageCanvas
from core.yolo_io import find_image_files, get_label_path
from core.yolo_seg_io import read_seg_labels, write_seg_labels, YoloSegAnnotation
from core.predictor import ModelManager, PredictionWorker


class EnterComboBox(QComboBox):
    enterPressed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().installEventFilter(self)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.enterPressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event):
        if obj == self.view() and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                idx = self.view().currentIndex()
                if idx.isValid():
                    self.setCurrentIndex(idx.row())
                self.hidePopup()
                self.enterPressed.emit()
                return True
        return super().eventFilter(obj, event)


class ClassEditDialog(QDialog):
    def __init__(self, class_names: List[str], current_id: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定類別")
        self.class_names = class_names.copy()
        if not self.class_names:
            self.class_names = ["object"]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"目前選中類別 ID: {current_id}"))

        self.combo = QComboBox()
        for i, name in enumerate(self.class_names):
            self.combo.addItem(f"{i}: {name}", i)
        if self.class_names:
            self.combo.setCurrentIndex(min(current_id, len(self.class_names) - 1))
        layout.addWidget(QLabel("選擇類別:"))
        layout.addWidget(self.combo)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("新增/修改類別名稱:", self.name_edit)
        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 999)
        self.id_spin.setValue(current_id)
        form.addRow("類別 ID:", self.id_spin)
        layout.addLayout(form)

        self.btn_apply_name = QPushButton("套用名稱變更")
        self.btn_apply_name.clicked.connect(self.apply_name)
        layout.addWidget(self.btn_apply_name)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_name(self):
        idx = self.id_spin.value()
        name = self.name_edit.text().strip()
        if not name:
            return
        while len(self.class_names) <= idx:
            self.class_names.append(f"class_{len(self.class_names)}")
        self.class_names[idx] = name
        self.combo.clear()
        for i, n in enumerate(self.class_names):
            self.combo.addItem(f"{i}: {n}", i)
        self.combo.setCurrentIndex(idx)
        self.name_edit.clear()

    def selected_class(self) -> int:
        return self.combo.currentData()


class MainWindow(QMainWindow):
    DEFAULT_CLASS_ID = 0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 標註工具 - Segmentation")
        self.resize(1400, 900)

        self.images_dir: Optional[Path] = None
        self.labels_dir: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.current_index: int = -1
        self.class_names: List[str] = []
        self.dirty = False

        self.model_manager = ModelManager()
        self.predict_worker: Optional[PredictionWorker] = None
        self._syncing_selected_class = False

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._update_button_states()

    # ==================== UI 建構 ====================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ---------- 左側：檔案清單 ----------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("📁 影像清單"))
        self.image_list = QListWidget()
        self.image_list.setMinimumWidth(220)
        left_layout.addWidget(self.image_list)
        self.path_label = QLabel("未載入資料夾")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #888; font-size: 11px;")
        left_layout.addWidget(self.path_label)
        left.setMinimumWidth(240)
        splitter.addWidget(left)

        # ---------- 中間：畫布 ----------
        self.canvas = ImageCanvas()
        self.canvas.setMinimumWidth(600)
        splitter.addWidget(self.canvas)

        # ---------- 右側：控制面板 ----------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right.setMinimumWidth(280)
        right.setMaximumWidth(360)

        # 類別設定
        cls_group = QGroupBox("類別設定")
        cls_layout = QVBoxLayout(cls_group)
        self.class_combo = QComboBox()
        self.class_combo.addItem("0: object", 0)
        cls_layout.addWidget(QLabel("繪製新框時的預設類別:"))
        cls_layout.addWidget(self.class_combo)
        self.btn_edit_class = QPushButton("編輯類別清單")
        cls_layout.addWidget(self.btn_edit_class)
        right_layout.addWidget(cls_group)

        # 標註資訊
        info_group = QGroupBox("標註資訊")
        info_layout = QVBoxLayout(info_group)
        self.lbl_image = QLabel("影像: -")
        self.lbl_class = QLabel("選中類別: -")
        self.lbl_coords = QLabel("座標: -")
        info_layout.addWidget(self.lbl_image)
        info_layout.addWidget(self.lbl_class)
        info_layout.addWidget(QLabel("更換選中標註類別："))
        self.selected_class_combo = EnterComboBox()
        self.selected_class_combo.setEnabled(False)
        self.selected_class_combo.setToolTip("選中標註框後，選擇類別並按 Enter 套用")
        info_layout.addWidget(self.selected_class_combo)
        hint_label = QLabel("選到類別後按 Enter 套用")
        hint_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(hint_label)
        info_layout.addWidget(self.lbl_coords)
        right_layout.addWidget(info_group)

        # 操作按鈕
        op_group = QGroupBox("標註操作")
        op_layout = QVBoxLayout(op_group)
        self.btn_delete = QPushButton("刪除選中標註 (Del)")
        self.btn_save = QPushButton("儲存標註 (Ctrl+S)")
        op_layout.addWidget(self.btn_delete)
        op_layout.addWidget(self.btn_save)
        right_layout.addWidget(op_group)

        # 模型預測
        model_group = QGroupBox("模型預測")
        model_layout = QVBoxLayout(model_group)
        self.lbl_model = QLabel("未載入模型")
        self.lbl_model.setWordWrap(True)
        model_layout.addWidget(self.lbl_model)
        self.btn_load_model = QPushButton("載入 YOLO 模型")
        model_layout.addWidget(self.btn_load_model)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Conf:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        conf_layout.addWidget(self.spin_conf)
        model_layout.addLayout(conf_layout)

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU:"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.01, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.45)
        iou_layout.addWidget(self.spin_iou)
        model_layout.addLayout(iou_layout)

        self.btn_predict = QPushButton("對目前影像預測")
        self.btn_predict_all = QPushButton("批次預測所有影像")
        model_layout.addWidget(self.btn_predict)
        model_layout.addWidget(self.btn_predict_all)
        right_layout.addWidget(model_group)

        # 視圖
        view_group = QGroupBox("視圖")
        view_layout = QVBoxLayout(view_group)
        self.btn_fit = QPushButton("適應視窗 (F)")
        self.btn_100 = QPushButton("實際大小 (1:1)")
        view_layout.addWidget(self.btn_fit)
        view_layout.addWidget(self.btn_100)
        right_layout.addWidget(view_group)

        right_layout.addStretch()
        splitter.addWidget(right)
        splitter.setSizes([260, 800, 320])

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就緒")

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("檔案(&F)")
        act_open = QAction("開啟影像資料夾", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_images_folder)
        file_menu.addAction(act_open)

        act_set_labels = QAction("設定 Labels 資料夾...", self)
        act_set_labels.triggered.connect(self.set_labels_folder)
        file_menu.addAction(act_set_labels)
        file_menu.addSeparator()

        act_save = QAction("儲存標註", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_current_labels)
        file_menu.addAction(act_save)

        act_save_all = QAction("全部儲存", self)
        act_save_all.setShortcut("Ctrl+Shift+S")
        act_save_all.triggered.connect(self.save_all_labels)
        file_menu.addAction(act_save_all)
        file_menu.addSeparator()

        act_quit = QAction("離開", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        nav_menu = menubar.addMenu("導覽(&N)")
        act_prev = QAction("上一張", self)
        act_prev.setShortcut("A")
        act_prev.triggered.connect(self.prev_image)
        nav_menu.addAction(act_prev)

        act_next = QAction("下一張", self)
        act_next.setShortcut("D")
        act_next.triggered.connect(self.next_image)
        nav_menu.addAction(act_next)

        tb = QToolBar("主工具列")
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)
        tb.addAction(act_open)
        tb.addAction(act_save)
        tb.addSeparator()
        tb.addAction(act_prev)
        tb.addAction(act_next)

    def _connect_signals(self):
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        self.canvas.box_changed.connect(self.on_box_changed)
        self.canvas.box_selected.connect(self.on_box_selected)
        self.btn_delete.clicked.connect(self.canvas.delete_selected)
        self.btn_save.clicked.connect(self.save_current_labels)
        self.btn_edit_class.clicked.connect(self.edit_classes)
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_predict.clicked.connect(self.predict_current)
        self.btn_predict_all.clicked.connect(self.predict_all)
        self.btn_fit.clicked.connect(self.fit_view)
        self.btn_100.clicked.connect(self.actual_size)
        self.class_combo.currentIndexChanged.connect(self.on_default_class_changed)
        self.selected_class_combo.enterPressed.connect(self.apply_selected_class_from_combo)

    # ==================== 資料夾操作 ====================
    def open_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇影像資料夾")
        if not folder:
            return
        self.images_dir = Path(folder)
        if self.labels_dir is None or not str(self.labels_dir).startswith(str(self.images_dir)):
            default_labels = self.images_dir.parent / "labels"
            self.labels_dir = default_labels if default_labels.exists() else None
        self.load_images()

    def set_labels_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇 Labels 資料夾")
        if folder:
            self.labels_dir = Path(folder)
            self.path_label.setText(f"影像: {self.images_dir}\nLabels: {self.labels_dir}")
            self.status.showMessage(f"Labels 資料夾: {self.labels_dir}")

    def load_images(self):
        if not self.images_dir:
            return
        self.image_paths = find_image_files(str(self.images_dir))
        self.image_list.clear()
        for p in self.image_paths:
            self.image_list.addItem(QListWidgetItem(str(p.relative_to(self.images_dir))))
        self.path_label.setText(
            f"影像: {self.images_dir}\nLabels: {self.labels_dir or '(與影像同資料夾)'}"
        )
        if self.image_paths:
            self.image_list.setCurrentRow(0)
        self._update_button_states()
        self.status.showMessage(f"載入 {len(self.image_paths)} 張影像")

    # ==================== 影像切換 ====================
    def on_image_selected(self, row: int):
        if row < 0 or row >= len(self.image_paths):
            return
        if self.dirty:
            ret = QMessageBox.question(
                self, "未儲存", "目前影像有未儲存的標註，是否儲存?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if ret == QMessageBox.Save:
                self.save_current_labels()
            elif ret == QMessageBox.Cancel:
                self.image_list.blockSignals(True)
                self.image_list.setCurrentRow(self.current_index)
                self.image_list.blockSignals(False)
                return

        self.current_index = row
        image_path = self.image_paths[row]

        if not self.canvas.load_image(str(image_path)):
            QMessageBox.warning(self, "錯誤", f"無法載入影像: {image_path}")
            return

        label_path = get_label_path(image_path, self.images_dir, self.labels_dir)
        annotations = read_seg_labels(label_path)

        self.canvas.set_annotations(annotations, self.class_names)
        self.dirty = False
        self.lbl_image.setText(f"影像: {image_path.name}")
        self.status.showMessage(f"[{row + 1}/{len(self.image_paths)}] {image_path.name}")
        self.clear_selected_class_ui()
        self._update_button_states()

    def prev_image(self):
        if self.current_index > 0:
            self.image_list.setCurrentRow(self.current_index - 1)

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.image_list.setCurrentRow(self.current_index + 1)

    # ==================== 標註管理 ====================
    def on_box_changed(self):
        self.dirty = True
        item = self.canvas.selected_item
        if item:
            if isinstance(item.annotation, YoloSegAnnotation):
                pts = item.annotation.points
                self.lbl_coords.setText(f"Polygon: {len(pts)} 個頂點")
            else:
                ann = item.annotation
                self.lbl_coords.setText(
                    f"cx={ann.x_center:.3f}, cy={ann.y_center:.3f}, "
                    f"w={ann.width:.3f}, h={ann.height:.3f}"
                )
        self._update_button_states()

    def on_box_selected(self, item):
        if item:
            self.lbl_class.setText(
                f"選中類別: {item.class_name} (id={item.annotation.class_id})"
            )
            self.refresh_selected_class_combo(item.annotation.class_id)
            self.btn_delete.setEnabled(True)
            self.on_box_changed()
        else:
            self.clear_selected_class_ui()
            self.btn_delete.setEnabled(False)

    def clear_selected_class_ui(self):
        self.lbl_class.setText("選中類別: -")
        self.lbl_coords.setText("座標: -")
        if hasattr(self, "selected_class_combo"):
            self.selected_class_combo.blockSignals(True)
            self.selected_class_combo.clear()
            self.selected_class_combo.setEnabled(False)
            self.selected_class_combo.blockSignals(False)

    def refresh_selected_class_combo(self, current_class_id: int):
        self._syncing_selected_class = True
        self.selected_class_combo.blockSignals(True)
        self.selected_class_combo.clear()
        if not self.class_names:
            self.class_names = ["object"]
        for i, name in enumerate(self.class_names):
            self.selected_class_combo.addItem(f"{i}: {name}", i)
        idx = self.selected_class_combo.findData(current_class_id)
        if idx < 0:
            self.selected_class_combo.addItem(
                f"{current_class_id}: class_{current_class_id}", current_class_id
            )
            idx = self.selected_class_combo.count() - 1
        self.selected_class_combo.setCurrentIndex(idx)
        self.selected_class_combo.setEnabled(True)
        self.selected_class_combo.blockSignals(False)
        self._syncing_selected_class = False

    def apply_selected_class_from_combo(self):
        if self._syncing_selected_class:
            return
        item = self.canvas.selected_item
        if not item:
            self.status.showMessage("請先選擇一個標註框")
            return
        idx = self.selected_class_combo.currentIndex()
        if idx < 0:
            return
        class_id = int(self.selected_class_combo.itemData(idx))
        self.canvas.set_selected_class(class_id, self.class_names)
        item = self.canvas.selected_item
        if item:
            self.lbl_class.setText(
                f"選中類別: {item.class_name} (id={item.annotation.class_id})"
            )
        self.dirty = True
        self.canvas.update()
        self._update_button_states()
        self.status.showMessage(f"已將標註類別改為: {item.class_name}，請記得儲存")

    def save_current_labels(self):
        if self.current_index < 0:
            return
        image_path = self.image_paths[self.current_index]
        label_path = get_label_path(image_path, self.images_dir, self.labels_dir)
        annotations = self.canvas.get_annotations()
        ok = write_seg_labels(label_path, [item.annotation for item in annotations])

        if ok:
            self.dirty = False
            self.status.showMessage(f"已儲存: {label_path}")
        else:
            QMessageBox.warning(self, "錯誤", f"無法寫入: {label_path}")
        self._update_button_states()

    def save_all_labels(self):
        self.save_current_labels()
        if not self.image_paths:
            return
        current = self.current_index
        for i, image_path in enumerate(self.image_paths):
            self.image_list.setCurrentRow(i)
            QApplication.processEvents()
            label_path = get_label_path(image_path, self.images_dir, self.labels_dir)
            annotations = self.canvas.get_annotations()
            write_seg_labels(label_path, [item.annotation for item in annotations])
        if current >= 0:
            self.image_list.setCurrentRow(current)
        self.status.showMessage("全部儲存完成")

    # ==================== 類別 ====================
    def edit_classes(self):
        current_id = self.DEFAULT_CLASS_ID
        if self.canvas.selected_item:
            current_id = self.canvas.selected_item.annotation.class_id
        dlg = ClassEditDialog(self.class_names, current_id, self)
        if dlg.exec_() == QDialog.Accepted:
            self.class_names = dlg.class_names
            if not self.class_names and self.model_manager.class_names:
                self.class_names = self.model_manager.class_names.copy()
            self._refresh_class_combo()
            self.canvas.set_annotations(self.canvas.get_annotations(), self.class_names)
            self.clear_selected_class_ui()
            self._update_button_states()

    def _refresh_class_combo(self):
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        if not self.class_names:
            self.class_names = ["object"]
        for i, name in enumerate(self.class_names):
            self.class_combo.addItem(f"{i}: {name}", i)
        self.class_combo.blockSignals(False)
        idx = self.class_combo.findData(self.DEFAULT_CLASS_ID)
        if idx >= 0:
            self.class_combo.setCurrentIndex(idx)

    def on_default_class_changed(self, idx):
        if idx >= 0:
            data = self.class_combo.itemData(idx)
            if data is not None:
                self.DEFAULT_CLASS_ID = int(data)

    def get_default_class(self):
        cls_id = self.class_combo.currentData()
        if cls_id is None:
            cls_id = 0
        cls_id = int(cls_id)
        cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
        return cls_id, cls_name

    # ==================== 模型 ====================
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 YOLO 模型",
            filter="Model Files (*.pt *.onnx);;All Files (*)"
        )
        if not path:
            return
        ok, msg = self.model_manager.load_ultralytics(path)
        if ok:
            self.lbl_model.setText(f"模型: {os.path.basename(path)}")
            if self.model_manager.class_names:
                self.class_names = self.model_manager.class_names.copy()
                self._refresh_class_combo()
                self.canvas.set_annotations(self.canvas.get_annotations(), self.class_names)

            if not self.model_manager.is_seg_model:
                msg += "（偵測模型：預測框將自動轉為可調整的矩形多邊形）"
            self.status.showMessage(msg)
        else:
            QMessageBox.critical(self, "錯誤", msg)

    @staticmethod
    def _pred_to_seg(ann_data) -> Optional[YoloSegAnnotation]:
        """將預測結果統一轉為 Segmentation 標註。
        偵測模型輸出的 BBox 會轉成可手動調整的矩形多邊形。"""
        cls_id = ann_data['class_id']
        if ann_data['type'] == 'poly':
            return YoloSegAnnotation(cls_id, ann_data['points'])
        cx, cy = ann_data['cx'], ann_data['cy']
        w, h = ann_data['width'], ann_data['height']
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return YoloSegAnnotation(cls_id, points)

    def predict_current(self):
        if not self.model_manager.is_loaded():
            QMessageBox.warning(self, "未載入模型", "請先載入 YOLO 模型")
            return
        if self.current_index < 0:
            return

        image_path = self.image_paths[self.current_index]
        self.status.showMessage("預測中...")
        self.btn_predict.setEnabled(False)

        self.predict_worker = PredictionWorker(
            self.model_manager.model,
            str(image_path),
            self.spin_conf.value(),
            self.spin_iou.value(),
            self.model_manager.model_type
        )
        self.predict_worker.finished.connect(self.on_predict_finished)
        self.predict_worker.error.connect(self.on_predict_error)
        self.predict_worker.start()

    def on_predict_finished(self, annotations):
        self.btn_predict.setEnabled(True)
        if not annotations:
            self.status.showMessage("未偵測到物件")
            return

        ret = QMessageBox.question(
            self, "預測完成",
            f"偵測到 {len(annotations)} 個物件。\n是否覆蓋現有標註?\n\nYes=覆蓋 No=合併",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if ret == QMessageBox.Cancel:
            return
        if ret == QMessageBox.Yes:
            self.canvas.clear_all_annotations()

        for ann_data in annotations:
            cls_id = ann_data['class_id']
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

            try:
                seg_ann = self._pred_to_seg(ann_data)
            except ValueError:
                continue
            if seg_ann is not None:
                self.canvas.add_annotation(seg_ann, cls_name)

        self.dirty = True
        self.status.showMessage(f"預測完成: {len(annotations)} 個標註")
        self._update_button_states()

    def on_predict_error(self, err):
        self.btn_predict.setEnabled(True)
        QMessageBox.critical(self, "預測錯誤", err)
        self.status.showMessage("預測失敗")

    def predict_all(self):
        if not self.model_manager.is_loaded():
            QMessageBox.warning(self, "未載入模型", "請先載入 YOLO 模型")
            return
        if not self.image_paths:
            return

        ret = QMessageBox.question(
            self, "批次預測",
            f"將對 {len(self.image_paths)} 張影像進行預測並儲存 labels。\n是否繼續?",
            QMessageBox.Yes | QMessageBox.No
        )
        if ret != QMessageBox.Yes:
            return

        self.btn_predict_all.setEnabled(False)
        original = self.current_index

        for i, image_path in enumerate(self.image_paths):
            self.status.showMessage(
                f"批次預測中 [{i + 1}/{len(self.image_paths)}]: {image_path.name}"
            )
            QApplication.processEvents()
            try:
                results = self.model_manager.model(
                    str(image_path),
                    conf=self.spin_conf.value(),
                    iou=self.spin_iou.value(),
                    verbose=False
                )
                annotations = []
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    if boxes is not None:
                        is_seg = hasattr(result, 'masks') and result.masks is not None
                        img_h, img_w = result.orig_shape
                        for j in range(len(boxes)):
                            cls_id = int(boxes.cls[j].item())
                            try:
                                if is_seg:
                                    poly_norm = result.masks.xyn[j].cpu().numpy()
                                    points = [(float(p[0]), float(p[1])) for p in poly_norm]
                                    annotations.append(YoloSegAnnotation(cls_id, points))
                                else:
                                    # 偵測模型：BBox 轉為矩形多邊形
                                    xyxy = boxes.xyxy[j].cpu().numpy()
                                    x1, y1, x2, y2 = (float(v) for v in xyxy)
                                    rect = [
                                        (x1 / img_w, y1 / img_h), (x2 / img_w, y1 / img_h),
                                        (x2 / img_w, y2 / img_h), (x1 / img_w, y2 / img_h),
                                    ]
                                    annotations.append(YoloSegAnnotation(cls_id, rect))
                            except ValueError:
                                pass

                label_path = get_label_path(image_path, self.images_dir, self.labels_dir)
                write_seg_labels(label_path, annotations)

            except Exception as e:
                print(f"Predict error for {image_path}: {e}")

        if original >= 0:
            self.image_list.setCurrentRow(original)

        self.btn_predict_all.setEnabled(True)
        self.status.showMessage(f"批次預測完成: {len(self.image_paths)} 張")
        QMessageBox.information(self, "完成", "批次預測完成")

    # ==================== 視圖 ====================
    def fit_view(self):
        self.canvas.fit_to_window = True
        self.canvas.fit_image()

    def actual_size(self):
        self.canvas.fit_to_window = False
        self.canvas.scale = 1.0
        self.canvas.center_image()

    def _update_button_states(self):
        has_image = self.current_index >= 0
        has_model = self.model_manager.is_loaded()
        self.btn_predict.setEnabled(has_image and has_model)
        self.btn_predict_all.setEnabled(has_model and bool(self.image_paths))
        self.btn_save.setEnabled(has_image)
        self.btn_delete.setEnabled(self.canvas.selected_item is not None)

    def closeEvent(self, event):
        if self.dirty:
            ret = QMessageBox.question(
                self, "未儲存", "有未儲存的標註，是否儲存後離開?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if ret == QMessageBox.Save:
                self.save_current_labels()
                event.accept()
            elif ret == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()