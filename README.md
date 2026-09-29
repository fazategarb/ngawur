# 🧠 NGAWUR
### Neural Gesture Analysis with Webcam-based User-input Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Real--time%20Vision-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)

**NGAWUR** adalah proyek sistem visi komputer (*Computer Vision*) dan *Deep Learning* untuk mendeteksi serta mengenali 8 variasi gestur tangan secara *real-time* menggunakan kamera webcam atau melalui unggahan gambar dengan pelabelan akurasi persentase (*confidence score*).

---

## 📂 Struktur Proyek (Anti-Spaghetti Architecture)

Struktur direktori disusun secara modular dan terpisah berdasarkan tanggung jawab masing-masing komponen:

```
ngawur/
├── .env                          # Konfigurasi Roboflow API
├── .gitignore                    # Pengabaian file build, venv, & weights besar
├── index.py                      # 🚀 CLI Launcher utama untuk menjalankan seluruh sistem
├── README.md                     # Dokumentasi panduan lengkap proyek
├── requirements.txt              # Daftar dependensi untuk demo lokal
│
├── dataset/                      # Dataset gestur tangan
│   └── hand-gesture.v2i.yolov8/  # Dataset format YOLOv8 (train, valid, test)
│
├── placeholder/                  # 📚 Template praktikum dosen (ANN & CNN)
│   ├── Praktikum_1_ANN_.ipynb
│   └── Praktikum_2_CNN.ipynb
│
├── notebooks/                    # 📓 NOTEBOOK PELATIHAN GOOGLE COLAB
│   └── NGAWUR_Training.ipynb     # Notebook lengkap (18 sel terstruktur & analisis kritis)
│
├── models/                       # 🤖 Bobot Model Deep Learning
│   ├── .gitkeep
│   └── best.pt                   # (Tempat meletakkan best.pt setelah selesai training)
│
├── demo/                         # 🎮 Aplikasi Demo
│   ├── webcam_demo.py            # Demo 1: OpenCV Real-time Webcam
│   └── app.py                    # Demo 2: Streamlit Web Dashboard (Upload & Export)
│
├── utils/                        # 🔧 Modul Utilitas Bersama
│   ├── __init__.py
│   └── inference.py              # Engine inferensi YOLOv8, palet warna, & visualizer
│
└── output/                       # 📁 Direktori penyimpanan snapshot & hasil export
```

---

## 🏷️ 8 Target Kelas Gestur Tangan

| No | Label Kelas | Deskripsi Singkat | Warna Bounding Box |
|---|---|---|---|
| 1 | `Help Me` | Sinyal darurat / butuh bantuan | Merah Terang |
| 2 | `Level 1` | Menunjukkan 1 jari | Cyan / Biru Laut |
| 3 | `Level 2` | Menunjukkan 2 jari (Peace) | Lime Green |
| 4 | `Level 3` | Menunjukkan 3 jari | Emas / Kuning |
| 5 | `Level 4` | Menunjukkan 4 jari | Ungu / Magenta |
| 6 | `Level 5` | Telapak tangan terbuka penuh (5 jari) | Deep Sky Blue |
| 7 | `Operating As Expected` | Gestur jempol / operasi normal | Emerald Green |
| 8 | `Turn-Down Operation` | Gestur jempol ke bawah / batalkan operasi | Oranye |

---

## ⚡ Panduan Menjalankan Proyek

### 1. Pelatihan Model di Google Colab (GPU Gratis)

Karena pelatihan model deep learning pada 10.000 gambar membutuhkan GPU yang mumpuni:

1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah notebook [notebooks/NGAWUR_Training.ipynb](notebooks/NGAWUR_Training.ipynb).
3. Pastikan Hardware Accelerator aktif pada GPU:
   - Pilih menu: `Runtime` ➔ `Change runtime type` ➔ Pilih **T4 GPU** ➔ `Save`.
4. Jalankan sel berurutan dari **CELL 1** sampai **CELL 18**:
   - Sel akan otomatis mengunduh dataset dari Roboflow API.
   - Melakukan Exploratory Data Analysis (EDA) dan augmentasi.
   - Melatih model YOLOv8n selama 50 epoch dengan *Early Stopping*.
   - Menghasilkan visualisasi *Loss Curves*, *Confusion Matrix*, *PR Curve*, dan evaluasi *Test Set*.
5. Pada **CELL 18**, file `best.pt` akan otomatis diunduh ke komputer Anda.
6. Pindahkan file `best.pt` tersebut ke folder `models/` pada repositori ini:
   ```
   ngawur/models/best.pt
   ```

---

### 2. Instalasi Dependensi Demo di Laptop Lokal

Pastikan Anda sudah mengaktifkan virtual environment:

```powershell
# Aktivasi virtual environment
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

---

### 3. Menjalankan Aplikasi Demo

Anda dapat menggunakan launcher interaktif `index.py` atau menjalankan modul demo secara langsung:

#### Opsi A: Launcher Interaktif (Direkomendasikan)
```powershell
python index.py
```

#### Opsi B: Demo 1 — Real-time Webcam (OpenCV)
```powershell
python demo/webcam_demo.py
```
* **Shortcut Kontrol Webcam**:
  - `Q` / `ESC` : Keluar
  - `S` : Simpan foto snapshot ke folder `output/`
  - `+` / `-` : Naikkan / turunkan ambang batas kepercayaan (*confidence threshold*)
  - `H` : Tampilkan / sembunyikan overlay bantuan HUD

#### Opsi C: Demo 2 — Web Dashboard Upload Gambar (Streamlit)
```powershell
streamlit run demo/app.py
```
* **Fitur Web App**:
  - Unggah foto gestur (*JPG, PNG, WebP*).
  - Pilih sampel uji langsung dari subfolder `dataset/test/images`.
  - Ambil foto instan via webcam browser.
  - Pengaturan *slider confidence* dan *IoU NMS*.
  - Tabel rincian metrik akurasi per gestur.
  - Tombol **Download Hasil Deteksi** (*PNG Export*).

---

## 📊 Metrik & Kriteria Evaluasi

- **Arsitektur**: YOLOv8n (*Anchor-Free Decoupled Head + CSPDarknet Backbone*)
- **Target mAP@0.5**: $\ge 90.0\%$
- **Regularisasi**: Early Stopping (*patience* = 15), Weight Decay ($0.0005$), AdamW Optimizer.
- **Dataset**: Roboflow Universe `hand-gesture-j3usj` v2 (10.000 gambar).

---

## 👨‍💻 Kontributor & Lisensi
- Dikembangkan untuk proyek Deep Learning / Computer Vision Praktikum.
- Dataset dilisensikan di bawah [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).