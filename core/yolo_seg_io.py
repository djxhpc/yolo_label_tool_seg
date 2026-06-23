# yolo_label_tool/core/yolo_seg_io.py
"""YOLO Segmentation 格式標註檔案讀寫模組"""
from pathlib import Path
from typing import List, Tuple, Optional, Sequence
import numpy as np

try:
    from .yolo_io import YoloAnnotation
except ImportError:
    from yolo_io import YoloAnnotation

Point = Tuple[float, float]

def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

class YoloSegAnnotation:
    """YOLO Segmentation 格式標註 (normalized 0-1)"""
    def __init__(self, class_id: int, points: Sequence[Point]):
        self.class_id = int(class_id)
        # 移除重複點並 clip 到 0-1
        cleaned = []
        for x, y in points:
            x, y = _clip01(x), _clip01(y)
            if not cleaned or abs(x - cleaned[-1][0]) > 1e-6 or abs(y - cleaned[-1][1]) > 1e-6:
                cleaned.append((x, y))
        
        if len(cleaned) < 3:
            raise ValueError("Polygon 至少需要 3 個點")
            
        # 確保首尾不相連 (YOLO 格式不儲存重複的首尾點)
        if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
            cleaned.pop()
            
        self.points: List[Point] = cleaned

    def to_line(self) -> str:
        coords = [f"{_clip01(p[0]):.6f} {_clip01(p[1]):.6f}" for p in self.points]
        return f"{self.class_id} " + " ".join(coords)

    @classmethod
    def from_line(cls, line: str) -> "YoloSegAnnotation":
        parts = line.strip().split()
        if len(parts) < 7:
            raise ValueError(f"Invalid YOLO seg line: {line}")
        
        class_id = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            raise ValueError("Coordinates must be pairs")
            
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        return cls(class_id, points)

    @classmethod
    def from_bbox(cls, bbox_ann: YoloAnnotation) -> "YoloSegAnnotation":
        """將 BBox 轉為矩形 Polygon"""
        xc, yc, w, h = bbox_ann.x_center, bbox_ann.y_center, bbox_ann.width, bbox_ann.height
        x1, y1 = _clip01(xc - w/2), _clip01(yc - h/2)
        x2, y2 = _clip01(xc + w/2), _clip01(yc + h/2)
        return cls(bbox_ann.class_id, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def to_bbox(self) -> YoloAnnotation:
        """將 Polygon 轉為外接 BBox"""
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return YoloAnnotation(self.class_id, (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)

def read_seg_labels(label_path: Path) -> List[YoloSegAnnotation]:
    annotations = []
    if not Path(label_path).exists(): return annotations
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: annotations.append(YoloSegAnnotation.from_line(line))
                except ValueError: continue
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
    return annotations

def write_seg_labels(label_path: Path, annotations: List[YoloSegAnnotation]) -> bool:
    try:
        Path(label_path).parent.mkdir(parents=True, exist_ok=True)
        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                f.write(ann.to_line() + '\n')
        return True
    except Exception as e:
        print(f"Error writing {label_path}: {e}")
        return False