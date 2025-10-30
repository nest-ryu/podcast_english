import io
import os
from pathlib import Path
import glob
import streamlit as st

# Reuse the processing utilities from the existing script
from smalltalk_auto_generator import (
    transcribe_audio_whisper,
    detect_dialogue_window_precise,
    make_pdf,
    ascii_safe,
)

import re


def derive_base_name(src: Path) -> str:
    """Build output base name like '75. paranoid' from source file name."""
    filename = src.stem
    episode_num = None
    title = None

    episode_match = re.search(r"[Ee]pisode\s*(\d+)", filename)
    if episode_match:
        episode_num = episode_match.group(1)
        title_match = re.search(r"^(.+?)\s*[-–]\s*[Ee]pisode", filename)
        if title_match:
            title = title_match.group(1).strip().lower()

    if episode_num is None:
        num_match = re.search(r"(\d+)", filename)
        if num_match:
            episode_num = num_match.group(1)
            title = re.sub(r"\d+", "", filename).strip().lower()
            title = re.sub(r"[^\w\s]", "", title).strip()

    if episode_num and title:
        return f"{episode_num}. {title}"
    return "스몰톡_회화부분"


def run_pipeline(src: Path, model_for_final: str = "base"):
    from pydub import AudioSegment

    st.write("Step 1: Extracting first 3 minutes…")
    audio = AudioSegment.from_file(src)
    three_min_audio = audio[: 3 * 60 * 1000]

    st.write("Step 2: Cutting rough section 40s→160s…")
    rough_dialogue = three_min_audio[40 * 1000 : 160 * 1000]
    tmp_rough = src.parent / "_st_tmp_rough.mp3"
    rough_dialogue.export(tmp_rough, format="mp3")

    st.write("Step 3: Quick transcription (tiny)…")
    tr_rough = transcribe_audio_whisper(tmp_rough, model_size="tiny")

    st.write("Step 4: Detecting precise window (music-aware)…")
    local_start, local_end = detect_dialogue_window_precise(
        tr_rough["segments"], tr_rough.get("duration") or 0.0
    )
    st.write(f"Local window: {local_start:.2f}s → {local_end:.2f}s")

    # Map back to original timeline (3-min slice starts at 0s of three_min_audio, but we cut 40s offset)
    actual_start = 40.0 + local_start
    actual_end = 40.0 + local_end
    st.write(f"Absolute window: {actual_start:.2f}s → {actual_end:.2f}s")

    precise_dialogue = three_min_audio[int(actual_start * 1000) : int(actual_end * 1000)]
    tmp_precise = src.parent / "_st_tmp_precise.mp3"
    precise_dialogue.export(tmp_precise, format="mp3")

    st.write(f"Step 5: Final transcription ({model_for_final})…")
    tr = transcribe_audio_whisper(tmp_precise, model_size=model_for_final)

    # Cleanup tmp files
    try:
        tmp_rough.unlink(missing_ok=True)
        tmp_precise.unlink(missing_ok=True)
    except Exception:
        pass

    # Build outputs
    base_name = derive_base_name(src)
    mp3_out = src.parent / f"{base_name}.mp3"
    precise_dialogue.export(mp3_out, format="mp3")

    eng_sentences = re.split(r"(?<=[.!?])\s+", tr.get("text", "").strip())
    eng_sentences = [s.strip() for s in eng_sentences if s.strip()]

    pdf_out = src.parent / f"{base_name}.pdf"
    pdf_title = ascii_safe(f"Episode {base_name.split('.')[0]}: {base_name.split('. ', 1)[-1].title()}")
    make_pdf(pdf_out, pdf_title, eng_sentences, [], [], [], "")

    st.success("Done.")
    st.write(f"MP3 saved: {mp3_out}")
    st.write(f"PDF saved: {pdf_out}")

    # Offer downloads
    with open(mp3_out, "rb") as f:
        st.download_button(
            label="Download MP3",
            data=f,
            file_name=mp3_out.name,
            mime="audio/mpeg",
        )
    with open(pdf_out, "rb") as f:
        st.download_button(
            label="Download PDF",
            data=f,
            file_name=pdf_out.name,
            mime="application/pdf",
        )


def main():
    st.set_page_config(page_title="Podcast Smalltalk Cutter", page_icon="🎧", layout="centered")
    st.title("Podcast Smalltalk Cutter 🎧")
    st.caption("Local file picker – no upload needed")

    default_dir = str(Path.cwd())
    root_dir = st.text_input("Folder to scan (absolute path)", value=default_dir)

    dir_path = Path(root_dir).expanduser().resolve()
    if not dir_path.exists():
        st.error("Folder does not exist.")
        return

    audio_files = sorted(
        [
            Path(p)
            for p in glob.glob(str(dir_path / "*.mp3"))
            + glob.glob(str(dir_path / "*.m4a"))
            + glob.glob(str(dir_path / "*.wav"))
        ]
    )

    if not audio_files:
        st.info("No audio files (.mp3/.m4a/.wav) found in this folder.")
        return

    file_labels = [f.name for f in audio_files]
    choice = st.selectbox("Select an audio file", file_labels, index=0)
    chosen = audio_files[file_labels.index(choice)]

    model_final = st.selectbox("Final transcription model", ["base", "small", "tiny"], index=0)

    if st.button("Run", type="primary"):
        run_pipeline(chosen, model_for_final=model_final)


if __name__ == "__main__":
    main()


