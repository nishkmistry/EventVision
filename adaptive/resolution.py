import cv2
import numpy as np
from typing import Tuple, List, Optional

def adapt_resolution(frame: np.ndarray, bbox: Optional[List[int]] = None, level: str = "LIGHT") -> Tuple[np.ndarray, Tuple[float, float]]:
    if frame is None or frame.size == 0:
        return frame, (1.0, 1.0)

    h, w = frame.shape[:2]

    if level == "SKIP":
        return frame, (1.0, 1.0)

    if level == "LIGHT":
        new_w, new_h = max(320, w // 2), max(180, h // 2)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, (w / float(new_w), h / float(new_h))

    if level == "STANDARD":
        return frame, (1.0, 1.0)

    if level == "HIGH_ACCURACY":
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w, x2 + pad_x)
            crop_y2 = min(h, y2 + pad_y)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size > 0:
                return crop, (1.0, 1.0)

        return frame, (1.0, 1.0)

    return frame, (1.0, 1.0)
