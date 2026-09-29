"""
============================================================
NGAWUR: Neural Gesture Analysis with Webcam-based User-input Recognition
Main Entry Point & Launcher
============================================================
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def print_banner():
    print("""
======================================================================
  🧠 NGAWUR : Neural Gesture Analysis with Webcam-based User-input
              Recognition
======================================================================
  Proyek Deep Learning Deteksi Gestur Tangan (YOLOv8 + OpenCV + Streamlit)
======================================================================
""")


def check_environment():
    """Memeriksa keberadaan model, dataset, dan konfigurasi environment."""
    print("🔍 Memeriksa Status Sistem NGAWUR:")

    # Cek Model
    model_path = ROOT_DIR / "models" / "best.pt"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Model weights lokal ditemukan: models/best.pt ({size_mb:.2f} MB)")
    else:
        print("  ⚠️  Model lokal 'models/best.pt' belum ditemukan.")
        print("     (Saran: Latih model di Google Colab via 'notebooks/NGAWUR_Training.ipynb' lalu simpan ke models/best.pt)")

    # Cek .env Roboflow API
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        print("  ✅ Konfigurasi .env / Roboflow API terdeteksi.")
    else:
        print("  ℹ️  File .env tidak ditemukan.")

    # Cek Notebook
    nb_path = ROOT_DIR / "notebooks" / "NGAWUR_Training.ipynb"
    if nb_path.exists():
        print(f"  ✅ Notebook Google Colab tersedia: {nb_path.name}")

    print("-" * 70)


def run_webcam():
    """Menjalankan Demo Real-time OpenCV Webcam."""
    webcam_script = ROOT_DIR / "demo" / "webcam_demo.py"
    print("\n🎥 Menjalankan OpenCV Webcam Demo...")
    subprocess.run([sys.executable, str(webcam_script)])


def run_streamlit():
    """Menjalankan Demo Streamlit Web App."""
    app_script = ROOT_DIR / "demo" / "app.py"
    print("\n🌐 Menjalankan Streamlit Web Dashboard...")
    subprocess.run(["streamlit", "run", str(app_script)])


def main():
    print_banner()
    check_environment()

    while True:
        print("\nSilakan pilih menu:")
        print("  [1] Jalankan Demo Webcam Real-time (OpenCV)")
        print("  [2] Jalankan Demo Web App Upload Gambar (Streamlit)")
        print("  [3] Cek Ulang Status Sistem")
        print("  [4] Keluar")

        try:
            choice = input("\nPilihan [1-4]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar...")
            break

        if choice == "1":
            run_webcam()
        elif choice == "2":
            run_streamlit()
        elif choice == "3":
            check_environment()
        elif choice == "4":
            print("Terima kasih! Sampai jumpa.")
            break
        else:
            print("Pilihan tidak valid. Masukkan angka 1 sampai 4.")


if __name__ == "__main__":
    main()
