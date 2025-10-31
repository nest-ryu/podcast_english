import matplotlib.pyplot as plt
from pydub import AudioSegment, silence
import numpy as np

# 🔹 파일 설정
SOURCE_FILE = "75. Paranoid.mp3"
OUTPUT_FILE = "75. Paranoid_dialogue_only.mp3"

# 1️⃣ 오디오 불러오기
audio = AudioSegment.from_file(SOURCE_FILE, format="mp3")

# 2️⃣ numpy 배열로 변환 (파형 분석용)
samples = np.array(audio.get_array_of_samples())
if audio.channels == 2:  # 스테레오라면 좌우 평균
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)
times = np.arange(len(samples)) / audio.frame_rate

# 3️⃣ 무음(또는 음악) 구간 감지
silence_ranges = silence.detect_silence(
    audio,
    min_silence_len=1200,           # 1.2초 이상 무음이면 감지
    silence_thresh=audio.dBFS - 20  # 평균 음압보다 20dB 낮으면 무음
)
silence_ranges = [(start/1000, end/1000) for start, end in silence_ranges]

# 4️⃣ 회화 구간 추정 (인트로/아웃트로 음악 제거)
if len(silence_ranges) >= 2:
    start_t = silence_ranges[0][1]
    end_t = silence_ranges[-1][0]
else:
    start_t = len(audio) * 0.1 / 1000
    end_t = len(audio) * 0.9 / 1000

print(f"🎯 회화 구간 추정: {start_t:.2f}초 ~ {end_t:.2f}초")

# 5️⃣ 파형 시각화
plt.figure(figsize=(12, 4))
plt.plot(times, samples, linewidth=0.5, color="gray")
plt.title("Audio waveform with silence detection")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# 무음 영역(빨간색)
for start, end in silence_ranges:
    plt.axvspan(start, end, color="red", alpha=0.2)

# 회화 추정 구간(초록색)
plt.axvspan(start_t, end_t, color="green", alpha=0.2, label="Detected Dialogue")
plt.legend()
plt.tight_layout()
plt.show()

# 6️⃣ 추출 및 저장
dialogue = audio[int(start_t*1000):int(end_t*1000)]
dialogue.export(OUTPUT_FILE, format="mp3")
print(f"✅ 회화 부분만 저장됨 → {OUTPUT_FILE}")
