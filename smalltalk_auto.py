import io
import os
from pathlib import Path
import glob
import streamlit as st
import zipfile

# === Minimal implementations to avoid circular imports ===
import re
import subprocess
import platform
import shutil

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


def detect_dialogue_with_silence(audio_segment):
    """Use pydub silence detection to find dialogue sections."""
    from pydub import silence
    # Detect silence ranges (1.2 seconds minimum, 20dB below average)
    silence_ranges = silence.detect_silence(
        audio_segment,
        min_silence_len=1200,  # 1.2 seconds
        silence_thresh=audio_segment.dBFS - 20  # 20dB below average
    )
    silence_ranges = [(start/1000, end/1000) for start, end in silence_ranges]
    
    # Extract dialogue section (after first silence, before last silence)
    if len(silence_ranges) >= 2:
        start_t = silence_ranges[0][1]  # End of first silence
        end_t = silence_ranges[-1][0]   # Start of last silence
    else:
        # Fallback: use 10% to 90% of audio
        start_t = len(audio_segment) * 0.1 / 1000
        end_t = len(audio_segment) * 0.9 / 1000
    
    return start_t, end_t

def refine_end_by_transcript(tr_result, min_words_per_seg: int = 3, tail_silence_threshold: float = 1.5) -> float | None:
    """Refine end time using Whisper segments.
    - Choose the end of the last segment with at least `min_words_per_seg` words.
    - If trailing tail (duration - last_end) >= tail_silence_threshold, return last_end.
    - Otherwise return None (keep original end).
    """
    try:
        segments = tr_result.get("segments", []) or []
        if not segments:
            return None
        def wc(t: str) -> int:
            return len(re.findall(r"\b\w+\b", t))
        speech_like = [s for s in segments if wc(s.get("text", "")) >= min_words_per_seg]
        last_end = (speech_like[-1]["end"] if speech_like else segments[-1]["end"]) or 0.0
        dur = tr_result.get("duration") or last_end
        if (dur - last_end) >= tail_silence_threshold:
            return float(last_end)
        return None
    except Exception:
        return None

def run_pipeline(src: Path, model_for_final: str = "base"):
    # Step 1: Cut rough section 40s→160s via ffmpeg
    st.write("Step 1: 40초~160초 구간 자르는 중…")
    tmp_rough = src.parent / "_st_tmp_rough.mp3"
    start_sec, end_sec = 40, 160
    duration = end_sec - start_sec
    try:
        # Resolve ffmpeg path (system or bundled via imageio-ffmpeg)
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            try:
                import imageio_ffmpeg as iioff
                ffmpeg_bin = iioff.get_ffmpeg_exe()
            except Exception:
                ffmpeg_bin = None
        if not ffmpeg_bin:
            st.error("ffmpeg not found. On web: add 'imageio-ffmpeg' to requirements. On server: install ffmpeg.")
            raise FileNotFoundError("ffmpeg not available")
        # Re-encode for compatibility
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(src),
            "-acodec", "libmp3lame", "-b:a", "192k",
            str(tmp_rough),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        st.error("ffmpeg cutting failed. Ensure ffmpeg is available (PATH or imageio-ffmpeg).")
        raise e

    # Step 2: Use pydub silence detection to find precise dialogue section
    st.write("Step 2: 무음 구간 감지로 회화 구간 추출 중…")
    try:
        from pydub import AudioSegment
        rough_audio = AudioSegment.from_file(str(tmp_rough), format="mp3")
        local_start, local_end = detect_dialogue_with_silence(rough_audio)
        st.write(f"   감지된 회화 구간: {local_start:.2f}초 ~ {local_end:.2f}초")
    except ImportError:
        st.warning("pydub not available, using full rough section")
        local_start, local_end = 0.0, duration
    
    # Step 3: Extract precise dialogue from rough section
    st.write("Step 3: 회화 부분만 추출 중…")
    tmp_precise = src.parent / "_st_tmp_precise.mp3"
    try:
        precise_dialogue = rough_audio[int(local_start*1000):int(local_end*1000)]
        precise_dialogue.export(str(tmp_precise), format="mp3")
    except (NameError, UnboundLocalError):
        # If pydub failed, use ffmpeg to cut from rough section
        precise_duration = local_end - local_start
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", str(local_start),
            "-t", str(precise_duration),
            "-i", str(tmp_rough),
            "-acodec", "libmp3lame", "-b:a", "192k",
            str(tmp_precise),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Step 4: Final transcription
    st.write(f"Step 4: 음성 인식 중 ({model_for_final})…")
    tr = transcribe_audio_whisper(tmp_precise, model_size=model_for_final)
    refined_end = refine_end_by_transcript(tr)
    if refined_end is not None:
        try:
            st.write(f"   끝부분 보정 적용: {refined_end:.2f}s 까지로 재컷팅…")
            tmp_precise_ref = src.parent / "_st_tmp_precise_refined.mp3"
            cmd2 = [
                ffmpeg_bin, "-y",
                "-i", str(tmp_precise),
                "-t", str(max(0.2, refined_end)),
                "-acodec", "libmp3lame", "-b:a", "192k",
                str(tmp_precise_ref),
            ]
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                tmp_precise.unlink(missing_ok=True)
            except Exception:
                pass
            tmp_precise = tmp_precise_ref
        except Exception:
            # If trimming fails, continue with original tmp_precise
            pass

    # Prepare final outputs
    base_name = derive_base_name(src)
    mp3_out = src.parent / f"{base_name}.mp3"
    
    # Save final MP3
    try:
        with open(tmp_precise, "rb") as rf, open(mp3_out, "wb") as wf:
            wf.write(rf.read())
    except Exception:
        pass

    # Cleanup tmp files
    for tmp_file in [tmp_rough, tmp_precise]:
        try:
            tmp_file.unlink(missing_ok=True)
        except Exception:
            pass

    # Step 5: Generate PDF
    st.write("Step 5: PDF 생성 중…")
    eng_sentences = re.split(r"(?<=[.!?])\s+", tr.get("text", "").strip())
    eng_sentences = [s.strip() for s in eng_sentences if s.strip()]

    pdf_out = src.parent / f"{base_name}.pdf"
    pdf_title = ascii_safe(f"Episode {base_name.split('.')[0]}: {base_name.split('. ', 1)[-1].title()}")
    make_pdf(pdf_out, pdf_title, eng_sentences, [], [], [], "")

    # Step 6: Create ZIP file with both MP3 and PDF
    zip_out = src.parent / f"{base_name}.zip"
    with zipfile.ZipFile(zip_out, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(mp3_out, mp3_out.name)
        zipf.write(pdf_out, pdf_out.name)
    
    # Read ZIP file for download
    with open(zip_out, "rb") as f:
        zip_bytes = f.read()

    # Show download button
    st.success("완료되었습니다. 다운로드 준비 완료!")
    with st.container(border=True):
        st.subheader("다운로드")
        st.download_button(
            label="전체 다운로드 (MP3 + PDF)",
            data=zip_bytes,
            file_name=zip_out.name,
            mime="application/zip",
            key="dl_all_inline",
            type="primary",
        )


def render_downloads():
    # no-op (kept for compatibility)
    return


def main():
    st.set_page_config(page_title="팟캐스트 회화 추출기", page_icon="🎧", layout="centered")
    st.title("팟캐스트 회화 추출기 🎧")
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


