import io
import os
from pathlib import Path
import glob
import streamlit as st

# === Minimal implementations to avoid circular imports ===
import re
import subprocess
import platform

def ascii_safe(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 .:-]", "", text)

def transcribe_audio_whisper(src_audio_path, model_size: str = "base"):
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(str(src_audio_path), fp16=False, word_timestamps=False)
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
    ]
    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "duration": result.get("duration", None),
    }

def detect_dialogue_window_precise(segments, total_duration):
    # Music-aware: long gaps (>=3s) where whisper produced no text
    def find_music_zones(seg_list, dur, threshold_s=3.0):
        zones = []
        prev_end = 0.0
        for s in seg_list:
            gap = s["start"] - prev_end
            if gap >= threshold_s:
                zones.append({"start": prev_end, "end": s["start"], "len": gap})
            prev_end = max(prev_end, s["end"])
        if dur and dur - prev_end >= threshold_s:
            zones.append({"start": prev_end, "end": dur, "len": dur - prev_end})
        return zones

    dur = total_duration or 0.0
    zones = find_music_zones(segments, dur, threshold_s=3.0)
    if zones:
        start_ts = zones[0]["end"]
        end_ts = dur
        if len(zones) >= 2:
            end_ts = zones[-1]["start"]
        if end_ts - start_ts < 10.0:
            end_ts = min(start_ts + 30.0, dur or start_ts + 30.0)
        if end_ts - start_ts > 180.0:
            end_ts = start_ts + 180.0
        return float(start_ts), float(end_ts)
    # Fallback: first 30s
    return 0.0, min(30.0, dur or 30.0)

def make_pdf(output_pdf, title, eng_sentences, _kor, _key, _summary, _mission):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    try:
        pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
        font_name = 'HYSMyeongJo-Medium'
    except Exception:
        font_name = 'Helvetica'
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, leftMargin=40, rightMargin=40, topMargin=48, bottomMargin=48)
    head = ParagraphStyle('head', fontName=font_name, fontSize=16, leading=22, spaceAfter=14)
    sub = ParagraphStyle('sub', fontName=font_name, fontSize=13, leading=19, spaceAfter=10)
    body = ParagraphStyle('body', fontName=font_name, fontSize=12, leading=18, spaceAfter=12)
    elems = []
    elems.append(Paragraph(ascii_safe(title), head))
    elems.append(Spacer(1, 16))
    elems.append(Paragraph("Transcript (English)", sub))
    for s in eng_sentences:
        elems.append(Paragraph(s, body))
    doc.build(elems)

import re


def derive_base_name(src: Path) -> str:
    """Build output base name like '75. paranoid' from source file name.
    Tries multiple patterns and falls back to a cleaned stem.
    """
    stem = src.stem

    # Normalize separators
    cleaned = stem.replace("|", "-").replace("_", " ")

    # 1) Common patterns: '... - Episode 75', 'Ep75', 'EP 75'
    m = re.search(r"(?i)\b(?:episode|ep)\s*(\d{1,4})\b", cleaned)
    if m:
        num = m.group(1)
        left = cleaned[: m.start()].strip()
        title = left or cleaned
        title = re.sub(r"[^A-Za-z0-9\s.-]", "", title).strip().lower()
        title = re.sub(r"\s+", " ", title)
        return f"{num}. {title}" if title else f"{num}. audio"

    # 2) Any first number in the name → use as episode, remainder as title
    m = re.search(r"(\d{1,4})", cleaned)
    if m:
        num = m.group(1)
        # Title: part before number or, if empty, after number
        before = cleaned[: m.start()].strip()
        after = cleaned[m.end() :].strip()
        candidate = before if before else after
        # Remove connectors like '-', '–'
        candidate = re.sub(r"^[\s\-–_|]+", "", candidate)
        title = re.sub(r"[^A-Za-z0-9\s.-]", "", candidate).strip().lower()
        title = re.sub(r"\s+", " ", title)
        return f"{num}. {title}" if title else f"{num}. audio"

    # 3) Fallback: cleaned lowercase stem as title only
    title = re.sub(r"[^A-Za-z0-9\s.-]", "", cleaned).strip().lower()
    title = re.sub(r"\s+", " ", title) or "audio"
    return title


def run_pipeline(src: Path, model_for_final: str = "base"):
    # Cut via ffmpeg to avoid pydub/audioop issues
    st.write("Step 1: Cutting fixed section 40s→160s…")
    tmp_precise = src.parent / "_st_tmp_precise.mp3"
    start_sec, end_sec = 40, 160
    duration = end_sec - start_sec
    try:
        # Re-encode for compatibility
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(src),
            "-acodec", "libmp3lame", "-b:a", "192k",
            str(tmp_precise),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        st.error("ffmpeg cutting failed. Ensure ffmpeg is installed and in PATH.")
        raise e

    st.write(f"Step 2: Final transcription ({model_for_final})…")
    tr = transcribe_audio_whisper(tmp_precise, model_size=model_for_final)

    # Prepare final outputs
    base_name = derive_base_name(src)
    mp3_out = src.parent / f"{base_name}.mp3"
    try:
        # Save/copy the cut audio as final MP3
        with open(tmp_precise, "rb") as rf, open(mp3_out, "wb") as wf:
            wf.write(rf.read())
    except Exception:
        pass

    # Cleanup tmp file
    try:
        tmp_precise.unlink(missing_ok=True)
    except Exception:
        pass

    eng_sentences = re.split(r"(?<=[.!?])\s+", tr.get("text", "").strip())
    eng_sentences = [s.strip() for s in eng_sentences if s.strip()]

    pdf_out = src.parent / f"{base_name}.pdf"
    pdf_title = ascii_safe(f"Episode {base_name.split('.')[0]}: {base_name.split('. ', 1)[-1].title()}")
    make_pdf(pdf_out, pdf_title, eng_sentences, [], [], [], "")

    # Persist results to session so clicks won't erase them on rerun
    with open(mp3_out, "rb") as f:
        mp3_bytes = f.read()
    with open(pdf_out, "rb") as f:
        pdf_bytes = f.read()

    # Show downloads inline (no session persistence, no rerun)
    st.success("Done. Downloads are ready below.")
    with st.container(border=True):
        st.subheader("Downloads")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download MP3",
                data=mp3_bytes,
                file_name=mp3_out.name,
                mime="audio/mpeg",
                key="dl_mp3_inline",
            )
        with col2:
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=pdf_out.name,
                mime="application/pdf",
                key="dl_pdf_inline",
            )


def render_downloads():
    # no-op (kept for compatibility)
    return


def main():
    st.set_page_config(page_title="Podcast Smalltalk Cutter", page_icon="🎧", layout="centered")
    st.title("Podcast Smalltalk Cutter 🎧")
    st.caption("Choose a local folder/file or upload an audio file, then click Run.")

    # Input mode (on web/Linux, disable Folder scan; default to Upload)
    available_modes = ["Upload", "Folder scan"] if platform.system() == "Windows" else ["Upload"]
    mode = st.radio("Input mode", available_modes, horizontal=True, index=0)
    if "Folder scan" not in available_modes:
        st.info("Folder scan is disabled on web deployments. Use Upload instead.")

    # Helper to persist an uploaded file to a temp path
    def _save_upload(uploaded) -> Path:
        # Preserve original file name to allow proper episode/title parsing
        orig_name = Path(uploaded.name).name
        # sanitize illegal path chars for Windows
        safe_name = re.sub(r'[\\/:*?"<>|]+', '_', orig_name)
        temp_path = Path.cwd() / safe_name
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        return temp_path

    # Downloads render inline after processing

    chosen_path: Path | None = None
    model_final = st.selectbox("Final transcription model", ["base", "small", "tiny"], index=0)

    if mode == "Folder scan":
        default_dir = str(Path.cwd())
        root_dir = st.text_input("Folder to scan (absolute path)", value=default_dir)

        dir_path = Path(root_dir).expanduser().resolve()
        if not dir_path.exists():
            st.info("Folder does not exist.")
        else:
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
            else:
                file_labels = [f.name for f in audio_files]
                choice = st.selectbox("Select an audio file", file_labels, index=0)
                chosen_path = audio_files[file_labels.index(choice)]

    else:  # Upload mode
        up = st.file_uploader("Upload audio file (.mp3/.m4a/.wav)", type=["mp3", "m4a", "wav"], accept_multiple_files=False)
        if up:
            chosen_path = _save_upload(up)
            st.success(f"Uploaded: {chosen_path.name}")

    if st.button("Run", type="primary"):
        if chosen_path and chosen_path.exists():
            run_pipeline(chosen_path, model_for_final=model_final)
        else:
            st.error("Please select a valid audio file first.")


if __name__ == "__main__":
    main()


