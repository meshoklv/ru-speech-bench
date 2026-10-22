# Windows wrapper: run the Qwen3-TTS benchmark and keep the log in a file
# (long ssh sessions drop; reading the log afterwards is more reliable).
# Usage (from the repo root, venv with qwen-tts activated):  .\tts\qwen3\run-bench.ps1 -Voice voiceA
param([string]$Voice = "voiceA", [string]$Log = ".\out\bench-qwen.log")
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
python .\tts\qwen3\qwen_bench.py --voice $Voice *>&1 | Out-File -FilePath $Log -Encoding utf8
"WRAPPERDONE" | Out-File -FilePath $Log -Encoding utf8 -Append
