"""
============================================================
DEMO 2: WEB APP GESTURE RECOGNITION (Streamlit)
Proyek: NGAWUR (Neural Gesture Analysis with Webcam-based User-input Recognition)
============================================================
Deskripsi:
Aplikasi web interaktif untuk mengunggah gambar gestur tangan,
menjalankan inferensi model deep learning YOLOv8, menampilkan label + persentase
akurasi, dan mengekspor gambar hasil deteksi.
============================================================
"""

import io
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Tambahkan root path proyek ke sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.inference import (
    CLASS_COLORS,
    CLASS_NAMES,
    GestureDetector,
    draw_detections,
    format_prediction_table,
)

# -------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# -------------------------------------------------------------
st.set_page_config(
    page_title="NGAWUR - Neural Gesture Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk tampilan modern & sleek
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #a0aec0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .badge-cls {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector(model_path: str, conf: float, iou: float):
    """Cache instance GestureDetector agar tidak reload ulang setiap interaksi."""
    return GestureDetector(
        model_path=model_path if model_path else None,
        conf_threshold=conf,
        iou_threshold=iou,
    )


def main():
    # ---------------------------------------------------------
    # Header Utama
    # ---------------------------------------------------------
    st.markdown('<div class="main-title">🧠 NGAWUR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title"><b>N</b>eural <b>G</b>esture <b>A</b>nalysis with <b>W</b>ebcam-based <b>U</b>ser-input <b>R</b>ecognition</div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # Sidebar: Kontrol & Pengaturan Model
    # ---------------------------------------------------------
    st.sidebar.header("⚙️ Pengaturan Model")

    # Pilihan sumber model
    models_dir = ROOT_DIR / "models"
    available_models = list(models_dir.glob("*.pt")) if models_dir.exists() else []

    model_options = {f"models/{p.name}": str(p) for p in available_models}
    model_options["Auto-Detect / Roboflow API"] = ""

    selected_model_label = st.sidebar.selectbox(
        "Pilih Model Weights (.pt):",
        options=list(model_options.keys()),
        index=0,
    )
    selected_model_path = model_options[selected_model_label]

    # Slider Threshold
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold (Min. Akurasi):",
        min_value=0.10,
        max_value=1.00,
        value=0.40,
        step=0.05,
        help="Objek hanya dideteksi jika skor kepastian model >= nilai ini.",
    )

    iou_thresh = st.sidebar.slider(
        "IoU Threshold (NMS):",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression untuk eliminasi bounding box duplikat.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🎨 Opsi Visualisasi")
    show_box = st.sidebar.checkbox("Tampilkan Bounding Box", value=True)
    show_label = st.sidebar.checkbox("Tampilkan Label Nama", value=True)
    show_conf = st.sidebar.checkbox("Tampilkan Persentase Akurasi", value=True)
    line_thickness = st.sidebar.slider("Ketebalan Garis:", min_value=1, max_value=6, value=2)

    # Inisialisasi Detector
    detector = load_detector(selected_model_path, conf_thresh, iou_thresh)
    st.sidebar.info(f"📌 **Status Backend:**\n{detector.model_source_desc}")

    # ---------------------------------------------------------
    # Daftar 8 Kelas Gestur
    # ---------------------------------------------------------
    with st.sidebar.expander("📋 8 Target Kelas Gestur", expanded=False):
        for name in CLASS_NAMES:
            color = CLASS_COLORS.get(name, (100, 100, 100))
            rgb_hex = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"
            st.markdown(
                f'<span class="badge-cls" style="background-color: {rgb_hex}; color: #ffffff;">{name}</span>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------
    # Tab Mode Input: Upload Gambar vs Ambil Foto Kamera
    # ---------------------------------------------------------
    tab_upload, tab_sample, tab_cam = st.tabs(
        ["📤 Upload Gambar", "🖼️ Sampel Test Dataset", "📸 Ambil Foto Webcam"]
    )

    pil_image = None
    source_name = "image"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Pilih file gambar untuk dianalisis (JPG, PNG, WebP):",
            type=["jpg", "jpeg", "png", "webp"],
            help="Unggah foto gestur tangan untuk dianalisis oleh model.",
        )
        if uploaded_file is not None:
            pil_image = Image.open(uploaded_file).convert("RGB")
            source_name = uploaded_file.name

    with tab_sample:
        test_dir = ROOT_DIR / "dataset" / "hand-gesture.v2i.yolov8" / "test" / "images"
        if test_dir.exists():
            sample_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
            if sample_files:
                selected_sample = st.selectbox(
                    "Pilih sampel dari test subfolder dataset:",
                    options=sample_files,
                    format_func=lambda p: p.name,
                )
                if st.button("Analisis Sampel Terpilih", use_container_width=True):
                    pil_image = Image.open(selected_sample).convert("RGB")
                    source_name = selected_sample.name
            else:
                st.warning("Tidak ada file gambar di folder test dataset.")
        else:
            st.info("Subfolder test dataset lokal tidak ditemukan. Silakan gunakan tab Upload.")

    with tab_cam:
        cam_photo = st.camera_input("Ambil foto gestur tangan langsung dari webcam:")
        if cam_photo is not None:
            pil_image = Image.open(cam_photo).convert("RGB")
            source_name = "webcam_capture.jpg"

    # ---------------------------------------------------------
    # Pemrosesan Inferensi dan Visualisasi Hasil
    # ---------------------------------------------------------
    if pil_image is not None:
        st.divider()

        # Konversi PIL ke format NumPy OpenCV (BGR)
        img_np_rgb = np.array(pil_image)
        img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)

        # Ukur waktu inferensi
        start_time = time.time()
        detections = detector.predict(img_np_bgr, conf_override=conf_thresh)
        inference_time_ms = (time.time() - start_time) * 1000

        # Visualisasi Bounding Box
        annotated_bgr = draw_detections(
            img_np_bgr,
            detections,
            show_box=show_box,
            show_label=show_label,
            show_conf=show_conf,
            line_thickness=line_thickness,
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # -----------------------------------------------------
        # Metric Summary Cards
        # -----------------------------------------------------
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{len(detections)}</div>
                    <div class="metric-lbl">Gestur Terdeteksi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            max_conf = (
                f"{max([d['confidence'] for d in detections]) * 100:.1f}%"
                if detections
                else "-"
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{max_conf}</div>
                    <div class="metric-lbl">Akurasi Tertinggi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            avg_conf = (
                f"{sum([d['confidence'] for d in detections]) / len(detections) * 100:.1f}%"
                if detections
                else "-"
            )
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{avg_conf}</div>
                    <div class="metric-lbl">Rata-rata Akurasi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-val">{inference_time_ms:.1f} ms</div>
                    <div class="metric-lbl">Latency Inferensi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # Tampilan Gambar (Before vs After)
        # -----------------------------------------------------
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("📷 Gambar Asli")
            st.image(pil_image, use_container_width=True)

        with col_img2:
            st.subheader("🎯 Hasil Deteksi Bounding Box")
            st.image(annotated_rgb, use_container_width=True)

        # -----------------------------------------------------
        # Tombol Download / Export Gambar
        # -----------------------------------------------------
        col_export, col_dummy = st.columns([1, 2])
        with col_export:
            # Encode hasil gambar annotated ke PNG buffer
            result_pil = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_img = buf.getvalue()

            export_filename = f"ngawur_annotated_{Path(source_name).stem}.png"
            st.download_button(
                label="📥 Download Hasil Deteksi (PNG)",
                data=byte_img,
                file_name=export_filename,
                mime="image/png",
                use_container_width=True,
            )

        # -----------------------------------------------------
        # Detail Tabel Prediksi
        # -----------------------------------------------------
        st.subheader("📊 Tabel Analisis Detail Prediksi")
        if detections:
            table_data = format_prediction_table(detections)
            st.dataframe(table_data, use_container_width=True)
        else:
            st.warning(
                "Tidak ada gestur yang terdeteksi dengan confidence >= threshold saat ini. "
                "Coba turunkan slider 'Confidence Threshold' di panel kiri."
            )
    else:
        st.info("👆 Silakan unggah gambar atau pilih sampel test dataset di atas untuk memulai analisis.")


if __name__ == "__main__":
    main()
