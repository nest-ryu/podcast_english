import os
import re
import math
import argparse
from pathlib import Path

# === Dependencies ===
# pip install openai-whisper pydub reportlab
# (ffmpeg required by pydub & whisper)
#
# Optional: For better speed on CPU-only environments, use:
# pip install faster-whisper

def transcribe_audio_whisper(src_audio_path, model_size="small"):
    """
    Transcribe using openai-whisper.
    Returns a dict with:
      - "text": full transcript string
      - "segments": list of {start, end, text}
      - "duration": audio length in seconds
    """
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(str(src_audio_path), fp16=False, word_timestamps=False)
    # Build a simple segments list
    segments = [{"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()} for seg in result.get("segments", [])]
    return {"text": result.get("text", "").strip(), "segments": segments, "duration": result.get("duration", None)}

def detect_dialogue_window(segments, total_duration):
    """
    Multi-strategy precise detection:
    1. Mark segments as MUSIC or SPEECH based on word count
    2. Find music→speech transitions (potential dialogue start)
    3. Verify with sentence density (dialogue = high density)
    4. Verify with short exchange pattern (dialogue = short sentences)
    5. Find speech→music transition (dialogue end)
    """
    def wlen(s): return len(re.findall(r"\b\w+\b", s))
    
    print(f"\n{'='*60}")
    print(f"Total segments: {len(segments)}")
    print(f"{'='*60}")
    
    # Mark each segment as MUSIC or SPEECH
    segment_types = []
    for seg in segments:
        word_count = wlen(seg["text"])
        seg_type = "MUSIC" if word_count < 2 else "SPEECH"
        segment_types.append({
            "start": seg["start"],
            "end": seg["end"],
            "type": seg_type,
            "words": word_count,
            "text": seg["text"]
        })
        print(f"[{seg['start']:6.1f}s] {seg_type:6s} ({word_count:2d}w) {seg['text'][:50]}")
    
    print(f"\n{'='*60}")
    print("ANALYZING SPEECH REGIONS...")
    print(f"{'='*60}\n")
    
    # Find all MUSIC → SPEECH transitions
    candidates = []
    for i in range(len(segment_types) - 1):
        if segment_types[i]["type"] == "MUSIC" and segment_types[i+1]["type"] == "SPEECH":
            candidate_start = segment_types[i+1]["start"]
            
            # Check next 5-10 segments for dialogue characteristics
            window_size = min(10, len(segments) - i - 1)
            window = segments[i+1:i+1+window_size]
            
            if len(window) >= 3:
                # Metric 1: Sentence density (sentences per second)
                time_span = window[-1]["end"] - window[0]["start"]
                density = len(window) / time_span if time_span > 0 else 0
                
                # Metric 2: Average sentence length (shorter = more dialogue-like)
                avg_words = sum(wlen(s["text"]) for s in window) / len(window)
                
                # Metric 3: Variance in length (dialogue has mixed short/medium)
                word_counts = [wlen(s["text"]) for s in window]
                variance = sum((w - avg_words)**2 for w in word_counts) / len(word_counts)
                
                candidates.append({
                    "start": candidate_start,
                    "density": density,
                    "avg_words": avg_words,
                    "variance": variance,
                    "sample": window[0]["text"][:40]
                })
                
                print(f"Candidate at {candidate_start:6.1f}s:")
                print(f"  Density: {density:.3f} sent/sec | Avg words: {avg_words:.1f} | Variance: {variance:.1f}")
                print(f"  Sample: '{window[0]['text'][:50]}'")
    
    # Select best candidate (high density + reasonable word length)
    start_ts = None
    if candidates:
        # Dialogue typically has: density > 0.15, avg_words between 4-15
        best_score = -1
        best_candidate = None
        
        for cand in candidates:
            # Score: prefer high density and medium word counts (not too short, not too long)
            density_score = min(cand["density"] * 10, 2.0)  # Cap at 2.0
            word_score = 1.0 if 4 <= cand["avg_words"] <= 15 else 0.3
            score = density_score * word_score
            
            print(f"\n  {cand['start']:6.1f}s => Score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_candidate = cand
        
        if best_candidate:
            start_ts = best_candidate["start"]
            print(f"\n✅ BEST DIALOGUE START: {start_ts:.2f}s (score: {best_score:.2f})")
            print(f"   '{best_candidate['sample']}'")

    # ---- Gap-based refinement (non-speech detection via segment gaps) ----
    def compute_long_gaps(seg_list, dur, threshold_s=2.5):
        gaps = []
        prev_end = 0.0
        for s in seg_list:
            if s["start"] - prev_end >= threshold_s:
                gaps.append({"start": prev_end, "end": s["start"], "len": s["start"] - prev_end})
            prev_end = max(prev_end, s["end"])
        if dur and dur - prev_end >= threshold_s:
            gaps.append({"start": prev_end, "end": dur, "len": dur - prev_end})
        return gaps

    long_gaps = compute_long_gaps(segments, total_duration or 0.0, threshold_s=2.5)

    # If no start found yet, choose the speech block between two long gaps with highest speech density
    if start_ts is None and len(long_gaps) >= 1:
        best_density = -1.0
        best_block = None
        for i in range(len(long_gaps) - 1):
            block_start = long_gaps[i]["end"]
            block_end = long_gaps[i+1]["start"]
            if block_end - block_start <= 5.0:
                continue
            # Compute speech time inside block
            speech_time = 0.0
            speech_segments = 0
            for s in segments:
                if s["end"] <= block_start or s["start"] >= block_end:
                    continue
                overlap = min(s["end"], block_end) - max(s["start"], block_start)
                if overlap > 0:
                    speech_time += overlap
                    speech_segments += 1
            duration = block_end - block_start
            density = (speech_time / duration) if duration > 0 else 0.0
            # Prefer blocks with reasonable number of segments as well
            if density > best_density and speech_segments >= 3:
                best_density = density
                best_block = (block_start, block_end)
        if best_block:
            start_ts = best_block[0]
            print(f"\n✅ GAP-BASED START: {start_ts:.2f}s (between long silences)")

    # Find end: if we have a start, end at next long gap or cap at +180s
    end_ts = None
    if start_ts is not None:
        # Next long gap after start
        next_gap_end = None
        for g in long_gaps:
            if g["start"] >= start_ts + 10.0:  # ensure at least some speech
                next_gap_end = g["start"]
                break
        if next_gap_end is not None:
            end_ts = next_gap_end
            print(f"✅ DIALOGUE END (next long silence): {end_ts:.2f}s")
        # Cap to 3 minutes from start
        if end_ts is None or end_ts - start_ts > 180.0:
            end_ts = (start_ts + 180.0)
            print(f"⏱️  Capped to 3 minutes at {end_ts:.2f}s")

    # Fallbacks
    if start_ts is None:
        start_ts = 0.0
        print("⚠️  Could not detect dialogue, using 0.0s\n")
    if end_ts is None:
        end_ts = min(start_ts + 180.0, total_duration if total_duration else 300.0)
        print(f"⚠️  Using 3-minute cap: {end_ts:.2f}s\n")
    
    print(f"{'='*60}")
    print(f"FINAL: {start_ts:.2f}s → {end_ts:.2f}s ({end_ts-start_ts:.1f}s duration)")
    print(f"{'='*60}\n")
    
    return float(start_ts), float(end_ts)

def detect_dialogue_window_precise(segments, total_duration):
    """
    Precise detector for music-bounded dialogue sections.
    Strategy: Long gaps (≥3s) = music/no speech zones
    Dialogue = between first and last long gap
    """
    print(f"\n{'='*60}")
    print(f"MUSIC-AWARE DETECTION")
    print(f"Total segments: {len(segments)}")
    print(f"{'='*60}\n")
    
    # Find LONG gaps (≥3s) = music zones where Whisper can't transcribe
    def find_music_zones(seg_list, dur, threshold_s=3.0):
        zones = []
        prev_end = 0.0
        for s in seg_list:
            gap_len = s["start"] - prev_end
            if gap_len >= threshold_s:
                zones.append({"start": prev_end, "end": s["start"], "len": gap_len})
            prev_end = max(prev_end, s["end"])
        if dur and dur - prev_end >= threshold_s:
            zones.append({"start": prev_end, "end": dur, "len": dur - prev_end})
        return zones
    
    music_zones = find_music_zones(segments, total_duration or 0.0, threshold_s=3.0)
    
    print(f"🎵 Found {len(music_zones)} music zones (gaps ≥3s):")
    for i, z in enumerate(music_zones):
        print(f"  Zone {i+1}: [{z['start']:6.1f}s - {z['end']:6.1f}s] ({z['len']:.1f}s)")
    
    # Strategy: Dialogue = after first music zone, before last music zone
    start_ts = 0.0
    end_ts = total_duration if total_duration else 180.0
    
    if len(music_zones) >= 1:
        # First music zone ends -> dialogue starts
        start_ts = music_zones[0]["end"]
        print(f"\n✅ Dialogue START: {start_ts:.2f}s (after first music zone)")
        
        if len(music_zones) >= 2:
            # Last music zone starts -> dialogue ends
            end_ts = music_zones[-1]["start"]
            print(f"✅ Dialogue END: {end_ts:.2f}s (before last music zone)")
        else:
            # Only one music zone (intro), use all remaining or cap at 3min
            end_ts = min(start_ts + 180.0, total_duration if total_duration else (start_ts + 180.0))
            print(f"✅ Dialogue END: {end_ts:.2f}s (single music zone, using 3min cap)")
    else:
        # No clear music zones, analyze segment distribution
        print("\n⚠️  No clear music zones found, using segment analysis...")
        
        if segments:
            # Use first and last segments with some padding
            start_ts = max(0.0, segments[0]["start"] - 1.0)
            end_ts = min(segments[-1]["end"] + 1.0, total_duration if total_duration else 180.0)
            print(f"✅ Using segment boundaries: {start_ts:.2f}s → {end_ts:.2f}s")
    
    # Ensure minimum dialogue length
    if end_ts - start_ts < 10.0:
        print(f"⚠️  Dialogue too short ({end_ts - start_ts:.1f}s), extending to 30s")
        end_ts = min(start_ts + 30.0, total_duration if total_duration else (start_ts + 30.0))
    
    # Cap at 3 minutes
    if end_ts - start_ts > 180.0:
        end_ts = start_ts + 180.0
        print(f"⏱️  Capped to 3 minutes")
    
    print(f"\n{'='*60}")
    print(f"✅ FINAL: {start_ts:.2f}s → {end_ts:.2f}s ({end_ts - start_ts:.1f}s)")
    print(f"{'='*60}\n")
    
    return float(start_ts), float(end_ts)

def detect_window_around_hint(segments, total_duration, hint_time, silence_threshold=2.5, max_clip_len=180.0):
    """
    Focused detector around a user-provided time.
    - Find the nearest long non-speech gaps (>= silence_threshold seconds)
      immediately before and after the hint time.
    - Use the gap end as start and the next gap start as end.
    - Cap duration to max_clip_len.
    """
    def gaps_from_segments(seg_list, dur, threshold_s):
        gaps = []
        prev_end = 0.0
        for s in seg_list:
            if s["start"] - prev_end >= threshold_s:
                gaps.append({"start": prev_end, "end": s["start"], "len": s["start"] - prev_end})
            prev_end = max(prev_end, s["end"])
        if dur and dur - prev_end >= threshold_s:
            gaps.append({"start": prev_end, "end": dur, "len": dur - prev_end})
        return gaps

    gaps = gaps_from_segments(segments, total_duration or 0.0, silence_threshold)

    # Find previous gap whose end <= hint_time and next gap whose start >= hint_time
    prev_gap = None
    next_gap = None
    for g in gaps:
        if g["end"] <= hint_time:
            if prev_gap is None or g["end"] > prev_gap["end"]:
                prev_gap = g
        if g["start"] >= hint_time and next_gap is None:
            next_gap = g
            break

    # Determine start/end
    start_ts = prev_gap["end"] if prev_gap else max(0.0, hint_time - 5.0)
    if next_gap:
        end_ts = next_gap["start"]
    else:
        end_ts = min(start_ts + max_clip_len, (total_duration or (start_ts + max_clip_len)))

    # Ensure sane bounds
    if end_ts - start_ts < 10.0:
        end_ts = min(start_ts + 60.0, (total_duration or (start_ts + 60.0)))

    # Cap to max clip length
    if end_ts - start_ts > max_clip_len:
        end_ts = start_ts + max_clip_len

    print(f"\n🎯 HINT-BASED SPLIT around {hint_time:.2f}s -> {start_ts:.2f}s to {end_ts:.2f}s")
    return float(start_ts), float(end_ts)

def export_dialogue_mp3(src_audio_path, dst_audio_path, start_ts, end_ts):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(src_audio_path)
    clip = audio[int(start_ts*1000): int(end_ts*1000)]
    clip.export(dst_audio_path, format="mp3")

def split_sentences(text):
    # Keep punctuation, split on end marks + space
    pieces = re.split(r'(?<=[.!?])\s+', text.strip())
    # Clean empties
    return [p.strip() for p in pieces if p.strip()]

def translate_ko(texts):
    """
    Simple translation stub using no external API to keep offline.
    Replace with actual translator for production (e.g., googletrans or your own glossary).
    """
    # Placeholder: return original text for now.
    # If you want auto-translate, uncomment below and install googletrans==4.0.0-rc1
    #
    # from googletrans import Translator
    # tr = Translator()
    # return [tr.translate(t, src='en', dest='ko').text for t in texts]
    return [t for t in texts]

def make_pdf(output_pdf, title, eng_sentences, kor_sentences, key_phrases, summary_points, daily_mission):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    # Register a CJK font for Korean safety
    try:
        pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
        font_name = 'HYSMyeongJo-Medium'
    except Exception:
        # Fallback to a generic font name; user should install a Unicode CID font if missing.
        font_name = 'Helvetica'

    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, leftMargin=40, rightMargin=40, topMargin=48, bottomMargin=48)

    body = ParagraphStyle(
        'body',
        fontName=font_name,
        fontSize=12,
        leading=18,       # comfortable line spacing
        spaceAfter=12,    # blank line between paragraphs
    )
    head = ParagraphStyle(
        'head',
        fontName=font_name,
        fontSize=16,
        leading=22,
        spaceAfter=14,
    )
    sub = ParagraphStyle(
        'sub',
        fontName=font_name,
        fontSize=13,
        leading=19,
        spaceAfter=10,
    )

    elems = []
    elems.append(Paragraph(title, head))
    elems.append(Spacer(1, 16))

    # English script only
    elems.append(Paragraph("Transcript (English)", sub))
    for s in eng_sentences:
        elems.append(Paragraph(s, body))

    doc.build(elems)

def ascii_safe(text):
    import re
    return re.sub(r'[^a-zA-Z0-9 .:-]', '', text)

def build(args):
    src = Path(args.source).resolve()
    assert src.exists(), f"Source audio not found: {src}"

    # Step 1: Extract first 3 minutes
    print("Step 1: Extracting first 3 minutes...")
    from pydub import AudioSegment
    audio = AudioSegment.from_file(src)
    three_min_audio = audio[:3*60*1000]  # 3 minutes in milliseconds
    
    # Step 2: Cut rough dialogue section: 40s ~ 2m40s (160s)
    print("Step 2: Cutting rough dialogue section (40s ~ 2m40s)...")
    rough_dialogue = three_min_audio[40*1000:160*1000]  # 40s to 160s
    
    temp_rough = src.parent / "temp_rough.mp3"
    rough_dialogue.export(temp_rough, format="mp3")
    
    # Step 3: Quick transcription to detect precise dialogue window
    print("Step 3: Quick transcription with tiny model to find precise dialogue...")
    tr_rough = transcribe_audio_whisper(temp_rough, model_size="tiny")
    
    # Step 4: Detect dialogue window within this rough section (focus on first 30s)
    print("Step 4: Detecting precise dialogue window (focused on first 30s)...")
    local_start, local_end = detect_dialogue_window_precise(tr_rough["segments"], tr_rough["duration"] or 0.0)
    
    print(f"   Found dialogue at {local_start:.2f}s ~ {local_end:.2f}s (within rough section)")
    
    # Step 5: Extract precise dialogue from original audio
    # Offset: rough section starts at 40s in the 3-min audio
    actual_start = 40.0 + local_start
    actual_end = 40.0 + local_end
    
    print(f"Step 5: Extracting precise dialogue from original audio...")
    print(f"   Absolute position: {actual_start:.2f}s ~ {actual_end:.2f}s")
    
    precise_dialogue = three_min_audio[int(actual_start*1000):int(actual_end*1000)]
    
    temp_precise = src.parent / "temp_precise.mp3"
    precise_dialogue.export(temp_precise, format="mp3")
    
    # Step 6: Final transcription with base model
    print("Step 6: Final transcription with base model...")
    tr = transcribe_audio_whisper(temp_precise, model_size="base")
    
    # Clean up temp files
    temp_rough.unlink()
    temp_precise.unlink()
    
    # Save transcription to file for analysis
    transcript_file = src.parent / "transcript_analysis.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("FULL TRANSCRIPTION (for analysis)\n")
        f.write("=" * 60 + "\n\n")
        for i, seg in enumerate(tr["segments"]):
            f.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] ({len(seg['text'].split())} words)\n")
            f.write(f"  {seg['text']}\n\n")
    
    print(f"\n✅ Transcription saved to: {transcript_file}")
    
    # Extract episode number and title from filename
    # Example: "Paranoid - Episode 75.mp3" -> "75. paranoid"
    filename = src.stem  # Remove extension
    episode_num = None
    title = None
    
    # Try to extract episode number
    episode_match = re.search(r'[Ee]pisode\s*(\d+)', filename)
    if episode_match:
        episode_num = episode_match.group(1)
        # Extract title (everything before "- Episode")
        title_match = re.search(r'^(.+?)\s*[-–]\s*[Ee]pisode', filename)
        if title_match:
            title = title_match.group(1).strip().lower()
    
    # Fallback: try to find any number
    if episode_num is None:
        num_match = re.search(r'(\d+)', filename)
        if num_match:
            episode_num = num_match.group(1)
            # Use whole filename as title (remove number and clean)
            title = re.sub(r'\d+', '', filename).strip().lower()
            title = re.sub(r'[^\w\s]', '', title).strip()
    
    # Generate output filenames
    if episode_num and title:
        base_name = f"{episode_num}. {title}"
    else:
        base_name = "스몰톡_회화부분"
    
    dst_mp3 = src.parent / f"{base_name}.mp3"
    print(f"\n✅ Saving precise dialogue MP3: {dst_mp3}")
    precise_dialogue.export(dst_mp3, format="mp3")

    # Prepare sentences
    eng_sentences = split_sentences(tr["text"])
    kor_sentences = translate_ko(eng_sentences)

    # Basic key phrases extraction (top n-grams by frequency)
    # Simple scoring: frequent short snippets (2-4 words)
    words = re.findall(r"[A-Za-z']+", " ".join(eng_sentences).lower())
    from collections import Counter
    def ngrams(words, n):
        return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    cand = Counter()
    for n in (2,3,4):
        cand.update(ngrams(words, n))
    key_phrases = [p for p,_ in cand.most_common(10) if len(p.split())>=2][:8]

    summary_points = [
        "인사 → 안부 → 자연스러운 주제 전환의 흐름을 파악합니다.",
        "상대의 정보를 확인하고 맞장구로 대화를 이어갑니다.",
        "마무리 멘트를 간결하게 정리합니다."
    ]

    daily_mission = "오늘 대화에서 배운 표현 3개를 선택해 실제로 말해보세요."

    # Build PDF with same naming format
    out_pdf = src.parent / f"{base_name}.pdf"
    title = f"Episode {episode_num}: {title.title()}" if episode_num and title else "비즈니스 스몰토크 회화 정리"
    title = ascii_safe(title)  # enforce clean ascii for PDF
    make_pdf(out_pdf, title, eng_sentences, kor_sentences, key_phrases, summary_points, daily_mission)

    print("Done.")
    print("PDF:", out_pdf)
    print("MP3:", dst_mp3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source MP3/M4A")
    parser.add_argument("--model",  type=str, default="base", help="whisper model size: tiny/base/small/medium/large")
    parser.add_argument("--start", type=float, default=None, help="Manual start time in seconds (e.g., 125.5)")
    parser.add_argument("--end", type=float, default=None, help="Manual end time in seconds (e.g., 180.0)")
    parser.add_argument("--around", type=float, default=None, help="Hint time in seconds; split at nearest long silences around it")
    parser.add_argument("--silence", type=float, default=2.5, help="Silence threshold seconds for around-mode (default 2.5s)")
    args = parser.parse_args()
    build(args)