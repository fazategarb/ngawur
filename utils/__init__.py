"""
NGAWUR Utilities Package
Neural Gesture Analysis with Webcam-based User-input Recognition
"""

from .inference import (
    CLASS_NAMES,
    CLASS_COLORS,
    GestureDetector,
    draw_detections,
    format_prediction_table,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_COLORS",
    "GestureDetector",
    "draw_detections",
    "format_prediction_table",
]
