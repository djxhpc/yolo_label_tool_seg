# yolo_label_tool/core/predictor.py
"""YOLO 模型預測模組 (支援 Detect & Segment)"""
import os
from typing import List, Tuple, Optional, Dict, Any
from PyQt5.QtCore import QThread, pyqtSignal

class PredictionWorker(QThread):
    # 回傳 list of dict: {'type': 'bbox'/'poly', 'class_id': int, 'conf': float, ...}
    finished = pyqtSignal(list) 
    error = pyqtSignal(str)

    def __init__(self, model, image_path: str, conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45, model_type: str = 'ultralytics'):
        super().__init__()
        self.model = model
        self.image_path = image_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_type = model_type

    def run(self):
        try:
            if self.model_type == 'ultralytics':
                results = self.model(
                    self.image_path, conf=self.conf_threshold, 
                    iou=self.iou_threshold, verbose=False
                )
                annotations = []
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None:
                        # 判斷是否為 Segmentation 模型
                        is_seg = hasattr(result, 'masks') and result.masks is not None
                        
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].item())
                            conf = float(boxes.conf[i].item())
                            
                            if is_seg:
                                # 取得 normalized polygon points
                                poly_norm = result.masks.xyn[i].cpu().numpy()
                                points = [(float(p[0]), float(p[1])) for p in poly_norm]
                                annotations.append({
                                    'type': 'poly', 'class_id': cls_id, 
                                    'conf': conf, 'points': points
                                })
                            else:
                                # BBox 邏輯
                                img_h, img_w = result.orig_shape
                                xyxy = boxes.xyxy[i].cpu().numpy()
                                x1, y1, x2, y2 = xyxy
                                annotations.append({
                                    'type': 'bbox', 'class_id': cls_id, 'conf': conf,
                                    'cx': ((x1 + x2) / 2.0) / img_w,
                                    'cy': ((y1 + y2) / 2.0) / img_h,
                                    'width': (x2 - x1) / img_w,
                                    'height': (y2 - y1) / img_h
                                })
                self.finished.emit(annotations)
            else:
                self.finished.emit([])
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class ModelManager:
    ULTRA = 'ultralytics'
    def __init__(self):
        self.model = None
        self.model_type: Optional[str] = None
        self.model_path: Optional[str] = None
        self.class_names: List[str] = []
        self.is_seg_model = False # 新增：標記是否為 seg 模型

    def load_ultralytics(self, model_path: str) -> Tuple[bool, str]:
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_type = self.ULTRA
            self.model_path = model_path
            
            # 判斷是否為 seg 模型 (透過 model task 或檔名)
            self.is_seg_model = getattr(self.model, 'task', '') == 'segment' or '-seg' in model_path
            
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = [self.model.names[i] for i in range(len(self.model.names))]
            return True, f"模型載入成功 ({'Seg' if self.is_seg_model else 'Detect'}): {os.path.basename(model_path)}"
        except Exception as e:
            return False, f"載入失敗: {e}"

    def is_loaded(self) -> bool:
        return self.model is not None