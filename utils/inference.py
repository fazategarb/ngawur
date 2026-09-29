"""
============================================================
MODULE: INFERENCE ENGINE & VISUALIZATION UTILITIES
Proyek: NGAWUR (Neural Gesture Analysis with Webcam-based User-input Recognition)
============================================================
Modul ini menangani:
1. Pemuatan model YOLOv8 (.pt) lokal atau fallback ke Roboflow Inference API.
2. Eksekusi inferensi pada frame gambar / video webcam.
3. Visualisasi bounding box estetik dengan label dan persentase akurasi (confidence).
4. Pemformatan data hasil deteksi untuk ditampilkan pada UI (Streamlit).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

# 8 Kelas Gestur Tangan NGAWUR
CLASS_NAMES = [
    "Help Me",
    "Level 1",
    "Level 2",
    "Level 3",
    "Level 4",
    "Level 5",
    "Operating As Expected",
    "Turn-Down Operation",
]

# Palet warna unik (BGR untuk OpenCV) untuk setiap kelas
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Help Me": (34, 34, 230),               # Merah Terang
    "Level 1": (225, 105, 65),              # Cyan / Biru Laut
    "Level 2": (50, 205, 50),               # Lime Green
    "Level 3": (0, 215, 255),               # Emas / Kuning
    "Level 4": (208, 102, 255),             # Ungu / Magenta
    "Level 5": (240, 160, 80),              # Deep Sky Blue
    "Operating As Expected": (60, 179, 113), # Emerald Green
    "Turn-Down Operation": (0, 140, 255),   # Oranye
}


class GestureDetector:
    """
    Kelas utama untuk melakukan deteksi gestur tangan.
    Mendukung model lokal YOLOv8 (.pt) dan fallback ke Roboflow Cloud Inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Inisialisasi detector.

        Args:
            model_path: Path file weights (.pt). Default mencari di models/best.pt
            conf_threshold: Ambang batas minimum confidence (0.0 - 1.0)
            iou_threshold: Ambang batas NMS IoU
            device: 'cpu', 'cuda', atau None (auto-detect)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.is_cloud = False
        self.model_source_desc = "Belum dimuat"

        # Tentukan default model path jika tidak dispesifikasikan
        if model_path is None:
            # Cari models/best.pt di direktori root proyek
            possible_paths = [
                Path("models/best.pt"),
                Path("../models/best.pt"),
                Path("best.pt"),
                Path("yolov8n.pt"),
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Memuat model lokal atau mengaktifkan mode cloud."""
        if self.model_path and Path(self.model_path).exists():
            try:
                from ultralytics import YOLO

                # Memuat model lokal YOLOv8
                self.model = YOLO(self.model_path)
                self.is_cloud = False
                self.model_source_desc = f"Model Lokal: {Path(self.model_path).name}"
                print(f"[INFO] Berhasil memuat model lokal: {self.model_path}")
                return
            except Exception as e:
                print(f"[WARNING] Gagal memuat model lokal ({e}). Beralih ke pengecekan Roboflow.")

        # Jika model lokal belum tersedia, cek apakah ada API Key di .env
        api_key = self._get_roboflow_api_key()
        if api_key:
            try:
                import importlib

                inf_module = importlib.import_module("inference_sdk")
                InferenceHTTPClient = getattr(inf_module, "InferenceHTTPClient")
                InferenceConfiguration = getattr(inf_module, "InferenceConfiguration")

                self.model = InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=api_key,
                ).configure(
                    InferenceConfiguration(api_key_transport="header")
                )
                self.is_cloud = True
                self.model_source_desc = "Roboflow Cloud API (hand-gesture-j3usj/2)"
                print("[INFO] Menggunakan Roboflow Cloud Inference API sebagai backend.")
                return
            except (ImportError, AttributeError, Exception) as e:
                print(f"[INFO] Backend inference_sdk tidak aktif / tidak terpasang ({e}).")

        self.model_source_desc = "Model belum tersedia (Latih di Colab / tambahkan best.pt)"
        print("[NOTICE] Model belum dimuat. Silakan letakkan weights di models/best.pt.")

    def _get_roboflow_api_key(self) -> Optional[str]:
        """Ekstrak API Key dari file .env jika ada."""
        env_paths = [Path(".env"), Path("../.env")]
        for env_path in env_paths:
            if env_path.exists():
                try:
                    content = env_path.read_text(encoding="utf-8")
                    for line in content.splitlines():
                        if "api_key=" in line:
                            # Parse api_key="KEY"
                            parts = line.split("api_key=")
                            if len(parts) > 1:
                                key = parts[1].split('"')[1].strip()
                                return key
                except Exception:
                    pass
        return None

    def predict(
        self,
        image_input: Union[np.ndarray, Image.Image, str, Path],
        conf_override: Optional[float] = None,
    ) -> List[Dict]:
        """
        Menjalankan prediksi deteksi gestur tangan pada gambar input.

        Args:
            image_input: NumPy array (BGR / RGB), PIL Image, atau Path gambar.
            conf_override: Opsional untuk override confidence threshold.

        Returns:
            List of dict berisi:
            [
                {
                    "class_name": str,
                    "confidence": float (0.0 - 1.0),
                    "confidence_pct": str ("95.4%"),
                    "box": [x1, y1, x2, y2] (integer koordinat piksel),
                    "color": (B, G, R)
                },
                ...
            ]
        """
        conf = conf_override if conf_override is not None else self.conf_threshold
        detections: List[Dict] = []

        if self.model is None:
            return detections

        # -------------------------------------------------------------
        # 1. Inferensi dengan Model Lokal Ultralytics YOLOv8
        # -------------------------------------------------------------
        if not self.is_cloud:
            results = self.model.predict(
                source=image_input,
                conf=conf,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    class_name = (
                        r.names[cls_id]
                        if r.names and cls_id in r.names
                        else (CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}")
                    )

                    color = CLASS_COLORS.get(class_name, (0, 255, 0))

                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": score,
                            "confidence_pct": f"{score * 100:.1f}%",
                            "box": xyxy,
                            "color": color,
                        }
                    )

        # -------------------------------------------------------------
        # 2. Inferensi via Roboflow Cloud API (Fallback)
        # -------------------------------------------------------------
        else:
            temp_path = None
            try:
                # Simpan sementara gambar jika inputnya numpy array / PIL
                if isinstance(image_input, (np.ndarray, Image.Image)):
                    temp_path = "_temp_infer.jpg"
                    if isinstance(image_input, Image.Image):
                        image_input.save(temp_path)
                    else:
                        cv2.imwrite(temp_path, image_input)
                    target_img = temp_path
                else:
                    target_img = str(image_input)

                # Panggil API
                res = self.model.infer(target_img, model_id="hand-gesture-j3usj/2")
                preds = res.get("predictions", [])

                for p in preds:
                    score = float(p.get("confidence", 0.0))
                    if score < conf:
                        continue

                    class_name = p.get("class", "Unknown")
                    cx = p.get("x", 0)
                    cy = p.get("y", 0)
                    w = p.get("width", 0)
                    h = p.get("height", 0)

                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2)
                    x2 = int(cx + w / 2)
                    y2 = int(cy + h / 2)

                    color = CLASS_COLORS.get(class_name, (0, 255, 0))

                    detections.append(
                        {
                            "class_id": -1,
                            "class_name": class_name,
                            "confidence": score,
                            "confidence_pct": f"{score * 100:.1f}%",
                            "box": [x1, y1, x2, y2],
                            "color": color,
                        }
                    )
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        return detections


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_box: bool = True,
    show_label: bool = True,
    show_conf: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Menggambar visualisasi bounding box, nama label, dan persentase akurasi
    dengan tampilan estetik dan kontras tinggi.

    Args:
        image: Frame gambar BGR (NumPy array).
        detections: Output dari GestureDetector.predict().
        show_box: Gambar kotak pembatas.
        show_label: Tampilkan nama gestur.
        show_conf: Tampilkan skor akurasi (%).
        line_thickness: Ketebalan garis.
        font_scale: Ukuran font label.

    Returns:
        NumPy array gambar BGR yang telah dianotasi.
    """
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        # Clamp koordinat ke batas dimensi gambar
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        color = det.get("color", (0, 255, 0))
        label_text = det["class_name"]

        if show_conf:
            label_text += f" ({det['confidence_pct']})"

        # 1. Gambar Bounding Box utama
        if show_box:
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                line_thickness,
                cv2.LINE_AA,
            )

            # Gambar corner accent (aksen sudut modern)
            corner_len = min(20, int((x2 - x1) * 0.25), int((y2 - y1) * 0.25))
            if corner_len > 5:
                accent_thickness = line_thickness + 2
                # Top-Left
                cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, accent_thickness)
                cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, accent_thickness)
                # Top-Right
                cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, accent_thickness)
                cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, accent_thickness)
                # Bottom-Left
                cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, accent_thickness)
                cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, accent_thickness)
                # Bottom-Right
                cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, accent_thickness)
                cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, accent_thickness)

        # 2. Gambar Badge Label dengan Background Solid
        if show_label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, font, font_scale, 1
            )

            # Posisi badge label (di atas kotak jika muat, atau di dalam kotak)
            badge_y1 = y1 - text_h - 10
            badge_y2 = y1
            if badge_y1 < 0:
                badge_y1 = y1
                badge_y2 = y1 + text_h + 10

            badge_x1 = x1
            badge_x2 = min(w_img, x1 + text_w + 14)

            # Background badge
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                color,
                cv2.FILLED,
            )

            # Teks label (putih dengan outline halus untuk readability optimal)
            text_x = badge_x1 + 6
            text_y = badge_y2 - 6
            cv2.putText(
                annotated,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated


def format_prediction_table(detections: List[Dict]) -> List[Dict]:
    """
    Format hasil deteksi menjadi format baris tabel untuk Streamlit DataFrame.
    """
    rows = []
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["box"]
        rows.append(
            {
                "No": i,
                "Gestur Terdeteksi": d["class_name"],
                "Akurasi (Confidence)": d["confidence_pct"],
                "Confidence Float": round(d["confidence"], 4),
                "Bounding Box (X1, Y1, X2, Y2)": f"({x1}, {y1}) → ({x2}, {y2})",
                "Lebar x Tinggi (px)": f"{max(0, x2 - x1)} × {max(0, y2 - y1)}",
            }
        )
    return rows
