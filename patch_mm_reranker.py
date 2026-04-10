with open("src/ai_workers/workers/mm_reranker.py") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "audio_data, sr = self._load_audio(query_audio_url)" in line:
        lines[i] = line.replace("sr", "_sr")
with open("src/ai_workers/workers/mm_reranker.py", "w") as f:
    f.writelines(lines)
