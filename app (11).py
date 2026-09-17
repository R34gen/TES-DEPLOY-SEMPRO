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

st.set_page_config(page_title="Pengenalan Aksara Jawa", layout="centered")

# ---------------------------------------------------------------------------
# Gaya & motion
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Inter:wght@400;500;600&family=Noto+Sans+Javanese&display=swap');

:root{
  --ink:#1E1A16; --muted:#8A7D6B; --line:#E4DCCD;
  --paper:#FAF7F0; --card:#FFFDF8; --accent:#6B4A2B; --accent-soft:#F1E8DA;
  --danger:#9A3B2E;
  --ease:cubic-bezier(.22,1,.36,1);        /* ease-out-quint */
}
html, body, .stApp{ font-family:'Inter', sans-serif; color:var(--ink); }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ padding-top:2.5rem; max-width:760px; }

/* --- Hilangkan kedip/redup bawaan Streamlit saat rerun --- */
[data-stale="true"], .stale-element{ opacity:1 !important; transition:none !important; }
[data-testid="stStatusWidget"]{ visibility:hidden; }

/* --- Keyframes --- */
@keyframes rise   { from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:none;} }
@keyframes pop    { from{opacity:0; transform:scale(.97);}       to{opacity:1; transform:none;} }
@keyframes fill   { from{transform:scaleX(0);}                   to{transform:scaleX(1);} }
@keyframes shimmer{ from{background-position:-400px 0;}          to{background-position:400px 0;} }

/* --- Header (masuk bertahap) --- */
.hero{ text-align:center; margin-bottom:1.6rem; }
.hero > *{ opacity:0; animation:rise .7s var(--ease) forwards; }
.hero .eyebrow{ animation-delay:.05s; font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; color:var(--muted); }
.hero .jawa{ animation-delay:.15s; font-family:'Noto Sans Javanese', serif; font-size:2.1rem; color:var(--accent); line-height:1.6; margin:.3rem 0 0; }
.hero h1{ animation-delay:.25s; font-family:'Source Serif 4', serif; font-weight:600; font-size:2rem; margin:.1rem 0 .5rem; padding:0; color:var(--ink); }
.hero p{ animation-delay:.35s; color:var(--muted); font-size:.93rem; margin:0; }

/* --- Indikator tahapan --- */
.steps{ display:flex; gap:.4rem; margin:0 0 1.4rem; }
.step{ flex:1; text-align:center; font-size:.78rem; color:var(--muted); transition:color .4s var(--ease); }
.step .bar{ position:relative; height:4px; border-radius:2px; background:var(--line); margin-bottom:.45rem; overflow:hidden; }
.step .bar i{ position:absolute; inset:0; background:var(--accent); transform:scaleX(0); transform-origin:left; }
.step.done{ color:var(--ink); font-weight:500; }
.step.done .bar i{ transform:scaleX(1); }
.step.done.fresh .bar i{ animation:fill .6s var(--ease) both; }
.step.done.fresh:nth-child(2) .bar i{ animation-delay:.12s; }
.step.done.fresh:nth-child(4) .bar i{ animation-delay:.12s; }

/* --- Kartu (masuk bertahap) --- */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--card); border-color:var(--line) !important; border-radius:10px;
  animation:rise .7s var(--ease) both; animation-delay:.4s;
  transition:box-shadow .3s var(--ease), border-color .3s var(--ease);
}
[data-testid="stVerticalBlockBorderWrapper"]:hover{ box-shadow:0 6px 24px rgba(30,26,22,.05); }
.label{ font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }

/* --- Uploader --- */
[data-testid="stFileUploaderDropzone"]{
  background:var(--paper); border:1.5px dashed #CFC3AE; border-radius:8px;
  transition:border-color .25s var(--ease), background .25s var(--ease);
}
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--accent); background:var(--accent-soft); }

/* --- Pratinjau --- */
.empty{ border:1.5px dashed var(--line); border-radius:8px; padding:3.2rem 1rem; text-align:center; color:var(--muted); font-size:.9rem; animation:pop .5s var(--ease) both; }
[data-testid="stImage"] img{ border-radius:6px; border:1px solid var(--line); animation:pop .55s var(--ease) both; }
.meta{ display:inline-block; font-size:.76rem; color:var(--muted); background:var(--paper); border:1px solid var(--line); border-radius:99px; padding:.15rem .7rem; margin-top:.3rem; animation:rise .5s var(--ease) both; animation-delay:.1s; }

/* --- Tombol --- */
.stButton > button{
  width:100%; border-radius:8px; padding:.65rem 0; font-weight:600; letter-spacing:.08em;
  transition:transform .2s var(--ease), box-shadow .25s var(--ease), background-color .25s var(--ease), opacity .25s;
}
.stButton > button:not(:disabled):hover{ transform:translateY(-1px); box-shadow:0 6px 18px rgba(107,74,43,.18); }
.stButton > button:not(:disabled):active{ transform:translateY(0) scale(.99); box-shadow:none; }

/* --- Hasil --- */
.result{ font-family:'Source Serif 4', serif; font-size:1.9rem; line-height:1.35; min-height:3.2rem; padding:.2rem 0; word-break:break-word; animation:rise .55s var(--ease) both; }
.result.placeholder{ font-family:'Inter', sans-serif; font-size:.95rem; color:var(--muted); font-style:italic; }
.result.error{ font-family:'Inter', sans-serif; font-size:.95rem; color:var(--danger); font-style:normal; }
.skeleton{ height:1.6rem; border-radius:6px; margin:.35rem 0;
  background:linear-gradient(90deg,var(--line) 0%,#F3EDE2 50%,var(--line) 100%);
  background-size:800px 100%; animation:shimmer 1.2s linear infinite; }
.skeleton.short{ width:55%; }

footer{ visibility:hidden; }

/* --- Aksesibilitas: hormati preferensi kurangi animasi --- */
@media (prefers-reduced-motion: reduce){
  *, *::before, *::after{ animation:none !important; transition:none !important; }
  .hero > *{ opacity:1; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
DEFAULTS = {"uploader_key": 0, "result": "", "note": "", "note_kind": "",
            "last_file": None, "processed": False, "prev_stage": 0}
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def clear_output() -> None:
    st.session_state.update(result="", note="", note_kind="", processed=False)


def reset() -> None:
    """Menghapus citra (dengan mengganti key uploader) dan hasil pengenalan."""
    st.session_state.uploader_key += 1
    st.session_state.last_file = None
    clear_output()
    st.toast("Tampilan dikosongkan", icon="↺")


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


def render_steps(slot, stage: int) -> None:
    """Indikator tahapan. Bar yang baru selesai diberi animasi pengisian."""
    names = ["Unggah", "Pratinjau", "Proses", "Hasil"]
    done_now = {0: 0, 1: 2, 2: 4}[stage]
    done_prev = {0: 0, 1: 2, 2: 4}[st.session_state.prev_stage]
    parts = []
    for i, name in enumerate(names):
        cls = "step"
        if i < done_now:
            cls += " done" + (" fresh" if i >= done_prev else "")
        parts.append(f"<div class='{cls}'><div class='bar'><i></i></div>{name}</div>")
    slot.markdown(f"<div class='steps'>{''.join(parts)}</div>", unsafe_allow_html=True)
    st.session_state.prev_stage = stage


def result_html(text: str, kind: str = "") -> str:
    return f"<div class='result {kind}'>{escape(text)}</div>"


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
                st.session_state.last_file = file_id
                clear_output()
                st.toast("Citra berhasil dimuat", icon="✅")
        except Exception:
            st.session_state.last_file = None
            st.toast("Berkas tidak dapat dibaca sebagai citra. Gunakan PNG atau JPG.", icon="⚠️")

    if image is not None:
        st.image(image)
        h, w = image.shape
        st.markdown(
            f"<span class='meta'>{escape(uploaded.name)} · {w} × {h} px · grayscale</span>",
            unsafe_allow_html=True,
        )
    elif st.session_state.last_file is None:
        st.markdown("<div class='empty'>Pratinjau citra akan tampil di sini</div>",
                    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. Tombol Proses
# ---------------------------------------------------------------------------
clicked = st.button("PROSES", type="primary", disabled=image is None)

# ---------------------------------------------------------------------------
# 4. Hasil Pengenalan (+ Notifikasi)
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.markdown("<div class='label'>Hasil Pengenalan Teks Latin</div>", unsafe_allow_html=True)
    out = st.empty()

    if clicked and image is not None:
        # Skeleton tampil selama inferensi berjalan (terlihat nyata setelah model dipasang)
        out.markdown("<div class='skeleton'></div><div class='skeleton short'></div>",
                     unsafe_allow_html=True)
        error = validate(image)
        if error:
            st.session_state.update(result="", note=error, note_kind="error", processed=False)
            st.toast(error, icon="⚠️")
        elif not CHECKPOINT.exists():
            st.session_state.update(result="", processed=True, note_kind="placeholder",
                                    note="Model belum tersedia pada tahap proposal. "
                                         "Hasil pengenalan akan tampil setelah model dilatih.")
            st.toast("Model belum tersedia pada tahap proposal", icon="ℹ️")
        else:
            st.session_state.update(result=recognize(image), processed=True,
                                    note="", note_kind="")

    if st.session_state.result:
        out.markdown(result_html(st.session_state.result), unsafe_allow_html=True)
    elif st.session_state.note:
        out.markdown(result_html(st.session_state.note, st.session_state.note_kind),
                     unsafe_allow_html=True)
    else:
        out.markdown(result_html("Hasil pengenalan akan tampil di sini", "placeholder"),
                     unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5. Reset
# ---------------------------------------------------------------------------
st.button("RESET", on_click=reset)

render_steps(steps_slot, 2 if st.session_state.processed else 1 if image is not None else 0)
