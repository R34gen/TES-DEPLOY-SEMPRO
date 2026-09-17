"""
Prototipe antarmuka Pengenalan Tulisan Tangan Aksara Jawa (tahap proposal).

Komponen mengikuti Tabel 3.5: Unggah Citra, Pratinjau Citra, Tombol Proses,
Hasil Pengenalan, Tombol Reset, dan Notifikasi. Indikator tahapan di bagian
atas mengikuti alur Gambar 3.3.

Pada tahap proposal model belum dilatih, sehingga `recognize()` belum memanggil
CNN-BiLSTM-CTC. Setelah checkpoint `best_val_cer.pt` tersedia, bagian bertanda
TODO diganti dengan pra-pemrosesan inferensi, model, dan greedy decoding yang
sama dengan kode evaluasi.
"""

from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

CHECKPOINT = Path("checkpoints/best_val_cer.pt")
MIN_HEIGHT = 16  # batas minimal tinggi citra (piksel)

st.set_page_config(page_title="Pengenalan Aksara Jawa", page_icon="ꦲ", layout="centered")

# ---------------------------------------------------------------------------
# Gaya: kertas gading, tinta gelap, satu aksen coklat sogan.
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600&family=Noto+Sans+Javanese&display=swap');

:root{
  --ink:#1E1A16; --muted:#8A7D6B; --line:#E4DCCD;
  --paper:#FAF7F0; --card:#FFFDF8; --accent:#6B4A2B; --accent-soft:#F1E8DA;
}
html, body, .stApp { font-family:'Inter', sans-serif; color:var(--ink); }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.5rem; max-width:760px; }

/* Header */
.hero{ text-align:center; margin-bottom:1.6rem; }
.hero .eyebrow{ font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:var(--muted); }
.hero .jawa{ font-family:'Noto Sans Javanese', serif; font-size:2.1rem; color:var(--accent); line-height:1.6; margin:.3rem 0 0; }
.hero h1{ font-family:'Source Serif 4', serif; font-weight:600; font-size:2rem; margin:.1rem 0 .5rem; padding:0; color:var(--ink); }
.hero p{ color:var(--muted); font-size:.93rem; margin:0; }

/* Indikator tahapan */
.steps{ display:flex; gap:.4rem; margin:0 0 1.4rem; }
.step{ flex:1; text-align:center; font-size:.78rem; color:var(--muted); }
.step .bar{ height:4px; border-radius:2px; background:var(--line); margin-bottom:.45rem; transition:background .35s; }
.step.done .bar{ background:var(--accent); }
.step.done{ color:var(--ink); font-weight:500; }
.step.now .bar{ background:linear-gradient(90deg,var(--accent) 50%,var(--line) 50%); }

/* Kartu */
[data-testid="stVerticalBlockBorderWrapper"]{ background:var(--card); border-color:var(--line) !important; border-radius:10px; }
.label{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }

/* Uploader */
[data-testid="stFileUploaderDropzone"]{ background:var(--paper); border:1.5px dashed #CFC3AE; border-radius:8px; transition:border-color .2s, background .2s; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent); background:var(--accent-soft); }

/* Pratinjau */
.empty{ border:1.5px dashed var(--line); border-radius:8px; padding:3.2rem 1rem; text-align:center; color:var(--muted); font-size:.9rem; }
.meta{ display:inline-block; font-size:.76rem; color:var(--muted); background:var(--paper); border:1px solid var(--line); border-radius:99px; padding:.15rem .7rem; margin-top:.3rem; }
[data-testid="stImage"] img{ border-radius:6px; border:1px solid var(--line); }

/* Tombol */
.stButton > button{ width:100%; border-radius:8px; padding:.65rem 0; font-weight:600; letter-spacing:.08em; transition:transform .12s, box-shadow .2s; }
.stButton > button:not(:disabled):hover{ transform:translateY(-1px); box-shadow:0 4px 14px rgba(107,74,43,.18); }

/* Hasil */
.result{ font-family:'Source Serif 4', serif; font-size:1.9rem; line-height:1.35; min-height:3.2rem; padding:.2rem 0; animation:rise .45s ease; word-break:break-word; }
.result.placeholder{ font-family:'Inter', sans-serif; font-size:.95rem; color:var(--muted); font-style:italic; }
@keyframes rise{ from{opacity:0; transform:translateY(6px);} to{opacity:1; transform:none;} }

footer{ visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, val in {"uploader_key": 0, "result": "", "processed": False,
                 "note": "", "last_file": None}.items():
    st.session_state.setdefault(key, val)


def reset() -> None:
    """Menghapus citra (dengan mengganti key uploader) dan hasil pengenalan."""
    st.session_state.uploader_key += 1
    st.session_state.update(result="", processed=False, note="", last_file=None)


def validate(image: np.ndarray) -> str | None:
    """Mengembalikan pesan galat bila citra tidak dapat diproses, atau None."""
    if image.shape[0] < MIN_HEIGHT:
        return "Resolusi citra terlalu kecil untuk dikenali."
    if image.std() < 2:
        return "Citra tampak kosong. Pastikan citra memuat tulisan tangan."
    return None


def recognize(image: np.ndarray) -> str:
    """Pengenalan satu citra. Saat ini placeholder tahap proposal."""
    # TODO (setelah training):
    # x = preprocess_eval(image)       # dari src/preprocessing.py
    # logits = model(x)                # CNN-BiLSTM-CTC, mode eval
    # return greedy_decode(logits)     # dari src/decoding.py
    return ""


def render_steps(stage: int) -> None:
    """Indikator tahapan: 0 kosong, 1 citra dimuat, 2 selesai diproses."""
    names = ["Unggah", "Pratinjau", "Proses", "Hasil"]
    done = {0: 0, 1: 2, 2: 4}[stage]
    html = "".join(
        f"<div class='step {'done' if i < done else 'now' if i == done else ''}'>"
        f"<div class='bar'></div>{n}</div>"
        for i, n in enumerate(names)
    )
    steps_slot.markdown(f"<div class='steps'>{html}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="eyebrow">Prototipe Sistem · CNN-BiLSTM-CTC</div>
  <div class="jawa">ꦲꦏ꧀ꦱꦫꦗꦮ</div>
  <h1>Pengenalan Tulisan Tangan Aksara Jawa</h1>
  <p>Unggah satu citra yang memuat satu jawaban tulisan tangan, bukan satu lembar jawaban penuh.</p>
</div>
""",
    unsafe_allow_html=True,
)
steps_slot = st.empty()  # diisi di akhir agar mencerminkan state terbaru

# ---------------------------------------------------------------------------
# 1–2. Unggah & Pratinjau Citra
# ---------------------------------------------------------------------------
image = None
with st.container(border=True):
    st.markdown("<div class='label'>Citra Masukan</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Unggah Citra Aksara Jawa",
        type=["png", "jpg", "jpeg"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_id = (uploaded.name, uploaded.size)
        try:
            image = np.array(Image.open(uploaded).convert("L"))
            if file_id != st.session_state.last_file:
                st.session_state.update(last_file=file_id, result="", processed=False, note="")
                st.toast("Citra berhasil dimuat", icon="✅")
        except Exception:
            st.error("Berkas tidak dapat dibaca sebagai citra. Gunakan PNG atau JPG.")

    if image is not None:
        st.image(image)
        h, w = image.shape
        st.markdown(
            f"<span class='meta'>{escape(uploaded.name)} · {w} × {h} px · grayscale</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='empty'>Pratinjau citra akan tampil di sini</div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. Proses
# ---------------------------------------------------------------------------
if st.button("PROSES", type="primary", disabled=image is None):
    with st.status("Memproses citra…", expanded=True) as status:
        st.write("Memeriksa citra masukan")
        error = validate(image)
        if error:
            status.update(label=error, state="error")
        elif not CHECKPOINT.exists():
            st.write("Memuat model CNN-BiLSTM-CTC")
            status.update(label="Model belum tersedia pada tahap proposal", state="complete",
                          expanded=False)
            st.session_state.update(processed=True, result="",
                                    note="Hasil pengenalan akan tampil setelah model dilatih.")
        else:
            st.write("Pra-pemrosesan, inferensi, dan greedy decoding")
            st.session_state.update(processed=True, result=recognize(image), note="")
            status.update(label="Pengenalan selesai", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# 4. Hasil Pengenalan
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("<div class='label'>Hasil Pengenalan Teks Latin</div>", unsafe_allow_html=True)
    if st.session_state.result:
        st.markdown(f"<div class='result'>{escape(st.session_state.result)}</div>",
                    unsafe_allow_html=True)
    else:
        text = st.session_state.note or "Hasil pengenalan akan tampil di sini"
        st.markdown(f"<div class='result placeholder'>{escape(text)}</div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. Reset
# ---------------------------------------------------------------------------
st.button("RESET", on_click=reset)

render_steps(2 if st.session_state.processed else 1 if image is not None else 0)
