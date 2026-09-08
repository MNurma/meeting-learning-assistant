import os
import tempfile

import streamlit as st
import whisper
import google.generativeai as genai


# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Meeting Learning Assistant",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Meeting Learning Assistant")
st.write(
    "Upload rekaman meeting, lalu ubah menjadi transkrip "
    "dan bahan belajar."
)


# =========================
# CEK API KEY GEMINI
# =========================

gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not gemini_key:
    st.error(
        "GEMINI_API_KEY belum ditemukan. "
        "Silakan set API key terlebih dahulu."
    )
    st.stop()

genai.configure(api_key=gemini_key)


# =========================
# LOAD MODEL WHISPER (sekali saja, di-cache)
# =========================

@st.cache_resource
def load_whisper_model():
    # Model "base" cukup ringan & cepat untuk laptop biasa.
    # Bisa diganti "small" atau "medium" kalau mau lebih akurat
    # (tapi lebih berat & lambat).
    return whisper.load_model("base")


# =========================
# UPLOAD AUDIO
# =========================

uploaded_file = st.file_uploader(
    "Upload rekaman meeting",
    type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
)


# =========================
# PROSES AUDIO
# =========================

if uploaded_file:

    st.audio(uploaded_file)

    st.write(f"**File:** {uploaded_file.name}")

    if st.button("🚀 Proses Audio"):

        try:
            with st.spinner("Memuat model transkripsi (Whisper)..."):
                model = load_whisper_model()

            with st.spinner("Sedang mentranskripsi audio... (bisa beberapa menit)"):

                # Simpan audio sementara
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_file_path = temp_file.name

                # Transkripsi pakai Whisper lokal
                result = model.transcribe(temp_file_path, language="id")
                transcript = result["text"]

                # Hapus file sementara
                os.remove(temp_file_path)

            st.success("Transkripsi berhasil!")

            st.subheader("📝 Transkrip")
            st.text_area(
                "Hasil transkripsi",
                transcript,
                height=300
            )

            # =========================
            # AI SUMMARY (Gemini)
            # =========================

            with st.spinner("Membuat kesimpulan dan bahan belajar..."):

                gemini_model = genai.GenerativeModel("gemini-3.6-flash")

                prompt = f"""
Kamu adalah asisten belajar untuk mahasiswa magang
yang sedang mempelajari materi teknikal.

Analisis transkrip meeting berikut.

Buat output dengan format:

1. Kesimpulan meeting
2. Poin-poin penting
3. Istilah teknikal dan penjelasan sederhananya
4. Hal yang perlu dipelajari lagi
5. Pertanyaan yang bisa ditanyakan ke mentor

Gunakan bahasa Indonesia yang natural,
jelas, dan mudah dipahami mahasiswa.

Jangan mengarang informasi yang tidak ada
di dalam transkrip.

TRANSKRIP:
{transcript}
"""

                response = gemini_model.generate_content(prompt)
                result_text = response.text

            st.subheader("📚 Bahan Belajar")

            st.markdown(result_text)

        except Exception as e:

            st.error(f"Terjadi error: {e}")