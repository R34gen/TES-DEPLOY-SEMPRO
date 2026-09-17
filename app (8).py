"""
Prototipe antarmuka Pengenalan Tulisan Tangan Aksara Jawa (tahap proposal).

Komponen mengikuti Tabel 3.5: Unggah Citra, Pratinjau Citra, Tombol Proses,
Hasil Pengenalan, Tombol Reset, dan Notifikasi.

Pada tahap proposal model belum dilatih, sehingga `recognize()` belum memanggil
CNN-BiLSTM-CTC. Setelah checkpoint `best_val_cer.pt` tersedia, bagian bertanda
TODO diganti dengan pra-pemrosesan inferensi, model, dan greedy decoding yang
sama dengan kode evaluasi.
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

CHECKPOINT = Path("checkpoints/best_val_cer.pt")
MIN_HEIGHT = 16  # batas minimal tinggi citra (piksel)

st.set_page_config(page_title="Pengenalan Aksara Jawa", layout="centered")

# ---------------------------------------------------------------------------
# Session state: Streamlit menjalankan ulang skrip setiap interaksi, sehingga
# hasil pengenalan dan kunci uploader disimpan agar Reset berfungsi.
# ---------------------------------------------------------------------------
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("result", "")
st.session_state.setdefault("last_file", None)


def reset() -> None:
    """Menghapus citra (dengan mengganti key uploader) dan hasil pengenalan."""
    st.session_state.uploader_key += 1
    st.session_state.result = ""
    st.session_state.last_file = None


def validate(image: np.ndarray) -> str | None:
    """Mengembalikan pesan galat bila citra tidak dapat diproses, atau None."""
    if image.shape[0] < MIN_HEIGHT:
        return "Resolusi citra terlalu kecil untuk dikenali."
    if image.std() < 2:  # citra hampir seragam / kosong
        return "Citra tampak kosong. Pastikan citra memuat tulisan tangan."
    return None


def recognize(image: np.ndarray) -> str:
    """Pengenalan satu citra. Saat ini placeholder tahap proposal."""
    # TODO (setelah training):
    # x = preprocess_eval(image)       # dari src/preprocessing.py
    # logits = model(x)                # CNN-BiLSTM-CTC, mode eval
    # return greedy_decode(logits)     # dari src/decoding.py
    return ""


# ---------------------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------------------
st.markdown(
    "<h3 style='text-align:center;letter-spacing:.04em'>"
    "PENGENALAN TULISAN TANGAN AKSARA JAWA</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Unggah satu citra yang memuat satu jawaban tulisan tangan Aksara Jawa "
    "(bukan satu lembar jawaban penuh)."
)

# 1. Unggah Citra
uploaded = st.file_uploader(
    "Unggah Citra Aksara Jawa",
    type=["png", "jpg", "jpeg"],
    key=f"uploader_{st.session_state.uploader_key}",
)

image = None
if uploaded is not None:
    # Citra baru -> hasil lama dihapus
    file_id = (uploaded.name, uploaded.size)
    if file_id != st.session_state.last_file:
        st.session_state.last_file = file_id
        st.session_state.result = ""
    try:
        image = np.array(Image.open(uploaded).convert("L"))  # grayscale
    except Exception:
        st.error("Berkas tidak dapat dibaca sebagai citra. Gunakan PNG atau JPG.")

# 2. Pratinjau Citra
if image is not None:
    st.image(image, caption="Pratinjau Citra")
else:
    st.markdown(
        "<div style='border:1px dashed #b8ab97;border-radius:4px;padding:56px 0;"
        "text-align:center;color:#8a7d6b'>Pratinjau citra akan tampil di sini</div>",
        unsafe_allow_html=True,
    )

st.write("")

# 3. Tombol Proses (nonaktif bila belum ada citra)
if st.button("PROSES", type="primary", disabled=image is None):
    error = validate(image)
    if error:
        st.warning(error)  # 6. Notifikasi
    elif not CHECKPOINT.exists():
        st.info(
            "Model belum tersedia pada tahap proposal. "
            "Hasil pengenalan akan tampil setelah model dilatih."
        )
    else:
        with st.spinner("Memproses citra..."):
            st.session_state.result = recognize(image)

# 4. Hasil Pengenalan
st.markdown("**Hasil Pengenalan Teks Latin**")
st.code(st.session_state.result or " ", language=None)

# 5. Tombol Reset
st.button("RESET", on_click=reset)
