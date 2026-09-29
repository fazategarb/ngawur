"""
============================================================
DEMO 1: REAL-TIME WEBCAM GESTURE RECOGNITION (OpenCV)
Proyek: NGAWUR (Neural Gesture Analysis with Webcam-based User-input Recognition)
============================================================
Deskripsi:
Aplikasi demo real-time deteksi gestur tangan via webcam menggunakan OpenCV & YOLOv8.
Menampilkan bounding box, label kelas, persentase akurasi, dan FPS counter secara interaktif.

Shortcut Keyboard:
  [Q] : Keluar dari aplikasi
  [S] : Simpan snapshot / screenshot frame saat ini
  [+] : Naikkan confidence threshold (+5%)
  [-] : Turunkan confidence threshold (-5%)
  [H] : Tampilkan / sembunyikan petunjuk bantuan
============================================================
"""

import os
import sys
import time
from pathlib import Path
import cv2

# Tambahkan root path proyek ke sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.inference import GestureDetector, draw_detections


def draw_hud(
    frame: cv2.Mat,
    fps: float,
    conf_thresh: float,
    model_desc: str,
    detection_count: int,
    show_help: bool = False,
) -> cv2.Mat:
    """
    Menggambar HUD (Heads-Up Display) status bar modern pada frame webcam.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Semi-transparent top status bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (20, 20, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Header Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "NGAWUR | Real-time Gesture Recognition"
    cv2.putText(annotated, title_text, (14, 26), font, 0.65, (0, 255, 200), 2, cv2.LINE_AA)

    # FPS & Metrics in top bar
    status_text = f"FPS: {fps:.1f} | Conf: {int(conf_thresh * 100)}% | Objek: {detection_count}"
    (tw, _), _ = cv2.getTextSize(status_text, font, 0.5, 1)
    cv2.putText(annotated, status_text, (w - tw - 14, 26), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    # Bottom status bar (model source & quick controls)
    overlay_bot = annotated.copy()
    cv2.rectangle(overlay_bot, (0, h - 30), (w, h), (15, 15, 20), cv2.FILLED)
    cv2.addWeighted(overlay_bot, 0.75, annotated, 0.25, 0, annotated)

    footer_text = f"Sumber: {model_desc} | Tekan [H] Bantuan | [S] Screenshot | [Q] Keluar"
    cv2.putText(annotated, footer_text, (14, h - 10), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Help overlay panel jika diaktifkan [H]
    if show_help:
        help_panel = annotated.copy()
        cv2.rectangle(help_panel, (w - 280, 50), (w - 10, 200), (25, 25, 30), cv2.FILLED)
        cv2.rectangle(help_panel, (w - 280, 50), (w - 10, 200), (0, 200, 255), 1)
        cv2.addWeighted(help_panel, 0.85, annotated, 0.15, 0, annotated)

        help_items = [
            "--- KONTROL CEPAT ---",
            "[Q] : Keluar",
            "[S] : Simpan Snapshot",
            "[+] : Naikkan Ambang (+5%)",
            "[-] : Turunkan Ambang (-5%)",
            "[H] : Toggle Bantuan",
        ]
        for i, item in enumerate(help_items):
            color = (0, 220, 255) if i == 0 else (220, 220, 220)
            cv2.putText(annotated, item, (w - 270, 75 + i * 20), font, 0.42, color, 1, cv2.LINE_AA)

    return annotated


def main():
    print("=" * 60)
    print("🚀 NGAWUR - Neural Gesture Analysis with Webcam Recognition")
    print("=" * 60)

    # Inisialisasi detector
    conf_threshold = 0.40
    detector = GestureDetector(conf_threshold=conf_threshold)

    print(f"[STATUS] Backend: {detector.model_source_desc}")

    # Buka webcam (index 0)
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka webcam pada indeks {camera_index}.")
        print("[INFO] Coba pastikan webcam tidak sedang digunakan oleh aplikasi lain.")
        return

    # Atur resolusi kamera default (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "NGAWUR - Real-time Hand Gesture Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # Inisialisasi variabel performa
    fps = 0.0
    prev_time = time.time()
    frame_counter = 0
    show_help = False

    # Buat direktori output jika belum ada
    output_dir = ROOT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[READY] Webcam aktif. Tekan [Q] pada jendela gambar untuk keluar.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Gagal membaca frame dari webcam.")
                break

            # Hitung FPS
            current_time = time.time()
            frame_counter += 1
            elapsed = current_time - prev_time
            if elapsed >= 0.5:  # Update FPS setiap 0.5 detik
                fps = frame_counter / elapsed
                frame_counter = 0
                prev_time = current_time

            # Jalankan prediksi gestur
            detections = detector.predict(frame, conf_override=conf_threshold)

            # Gambar bounding box dan label akurasi
            annotated_frame = draw_detections(
                frame,
                detections,
                show_box=True,
                show_label=True,
                show_conf=True,
                line_thickness=2,
            )

            # Gambar HUD Status Bar
            display_frame = draw_hud(
                annotated_frame,
                fps=fps,
                conf_thresh=conf_threshold,
                model_desc=detector.model_source_desc,
                detection_count=len(detections),
                show_help=show_help,
            )

            cv2.imshow(window_name, display_frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # Q atau ESC
                print("[INFO] Menutup aplikasi demo.")
                break

            elif key == ord("s"):  # Simpan screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = output_dir / f"ngawur_snap_{timestamp}.jpg"
                cv2.imwrite(str(save_path), display_frame)
                print(f"[SNAPSHOT] Berhasil disimpan di: {save_path}")

            elif key == ord("+") or key == ord("="):  # Naikkan confidence
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"[CONFIG] Confidence threshold dinaikkan ke: {conf_threshold:.2f}")

            elif key == ord("-") or key == ord("_"):  # Turunkan confidence
                conf_threshold = max(0.10, conf_threshold - 0.05)
                print(f"[CONFIG] Confidence threshold diturunkan ke: {conf_threshold:.2f}")

            elif key == ord("h"):  # Toggle help
                show_help = not show_help

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Resource webcam berhasil dilepaskan.")


if __name__ == "__main__":
    main()
