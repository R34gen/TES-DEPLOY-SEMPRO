"""
Prototipe antarmuka Pengenalan Tulisan Tangan Aksara Jawa (tahap proposal).

Komponen mengikuti Tabel 3.5: Unggah Citra, Pratinjau Citra, Tombol Proses,
Hasil Pengenalan, Tombol Reset, dan Notifikasi.

Pada tahap proposal model belum dilatih, sehingga fungsi `recognize()` belum
memanggil CNN-BiLSTM-CTC. Setelah checkpoint `best_val_cer.pt` tersedia,
bagian bertanda TODO diganti dengan pra-pemrosesan inferensi, model, dan
greedy decoding yang sama dengan kode evaluasi.
"""

from pathlib import Path

import gradio as gr
import numpy as np

CHECKPOINT = Path("checkpoints/best_val_cer.pt")
MIN_HEIGHT = 16  # batas minimal tinggi citra (piksel)


def validate(image: np.ndarray | None) -> str | None:
    """Mengembalikan pesan galat bila citra tidak dapat diproses, atau None."""
    if image is None:
        return "Silakan unggah satu citra jawaban Aksara Jawa terlebih dahulu."
    if image.shape[0] < MIN_HEIGHT:
        return "Resolusi citra terlalu kecil untuk dikenali."
    if image.std() < 2:  # citra hampir seragam / kosong
        return "Citra tampak kosong. Pastikan citra memuat tulisan tangan."
    return None


def recognize(image: np.ndarray | None) -> str:
    """Menjalankan validasi dan pengenalan untuk satu citra."""
    error = validate(image)
    if error:
        gr.Warning(error)
        return ""

    if not CHECKPOINT.exists():
        gr.Info("Model belum tersedia pada tahap proposal. "
                "Hasil pengenalan akan tampil setelah model dilatih.")
        return ""

    # TODO (setelah training):
    # x = preprocess_eval(image)          # dari src/preprocessing.py
    # logits = model(x)                   # CNN-BiLSTM-CTC, mode eval
    # return greedy_decode(logits)        # dari src/decoding.py
    return ""


def on_image_change(image):
    """Tombol Proses hanya aktif bila citra sudah dipilih."""
    return gr.update(interactive=image is not None), ""


def reset():
    """Menghapus citra dan hasil pengenalan."""
    return None, "", gr.update(interactive=False)


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
)

css = """
.gradio-container {max-width: 720px !important; margin: auto;}
#title {text-align: center; letter-spacing: .04em;}
#result textarea {font-size: 26px !important; line-height: 1.4;}
#btn-process {background: #3b2f25; color: #fff;}
footer {display: none !important;}
"""

with gr.Blocks(theme=theme, css=css, title="Pengenalan Aksara Jawa") as demo:
    gr.Markdown("## PENGENALAN TULISAN TANGAN AKSARA JAWA", elem_id="title")
    gr.Markdown(
        "Unggah satu citra yang memuat **satu jawaban** tulisan tangan "
        "Aksara Jawa (bukan satu lembar jawaban penuh)."
    )

    image = gr.Image(
        label="Pratinjau Citra",
        type="numpy",
        image_mode="L",               # langsung grayscale
        sources=["upload", "clipboard"],
        height=260,
    )
    btn_process = gr.Button("PROSES", interactive=False, elem_id="btn-process")
    result = gr.Textbox(
        label="Hasil Pengenalan Teks Latin",
        lines=2,
        interactive=False,
        show_copy_button=True,
        elem_id="result",
    )
    btn_reset = gr.Button("RESET", variant="secondary")

    image.change(on_image_change, inputs=image, outputs=[btn_process, result])
    btn_process.click(recognize, inputs=image, outputs=result)
    btn_reset.click(reset, outputs=[image, result, btn_process])

if __name__ == "__main__":
    demo.launch()
