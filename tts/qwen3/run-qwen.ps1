# Windows wrapper: run the Qwen3-TTS phrase synthesis and keep the log in a file.
# Usage (from the repo root, venv with qwen-tts activated):  .\tts\qwen3\run-qwen.ps1
param([string]$Log = ".\out\synth-qwen.log")
New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUNBUFFERED = "1"
python .\tts\qwen3\qwen_synth.py --out .\out\qwen3 *>&1 | Out-File -FilePath $Log -Encoding utf8
"EXITCODE=$LASTEXITCODE" | Out-File -FilePath $Log -Encoding utf8 -Append
"WRAPPERDONE" | Out-File -FilePath $Log -Encoding utf8 -Append
