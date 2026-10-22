# Resumable download of the ESpeech-TTS-1 RL-V2 weights (HTTP Range requests, retries).
# Useful on slow/flaky links where huggingface-cli stalls.
# Usage:  .\dl-espeech.ps1 [-Dir .\models]
param([string]$Dir = ".\models")
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
$targets = @(
  @{ url = "https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2/resolve/main/espeech_tts_rlv2.pt"; out = "$Dir\espeech_tts_rlv2.pt"; size = 2697096939 },
  @{ url = "https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2/resolve/main/vocab.txt";           out = "$Dir\vocab.txt";           size = 0 }
)
foreach ($t in $targets) {
  $attempt = 0
  while ($attempt -lt 60) {
    $attempt++
    $have = if (Test-Path $t.out) { (Get-Item $t.out).Length } else { 0 }
    if ($t.size -gt 0 -and $have -ge $t.size) { Write-Output "DONE $($t.out) $have"; break }
    if ($t.size -eq 0 -and $have -gt 0) { Write-Output "DONE $($t.out) $have"; break }
    try {
      $req = [Net.HttpWebRequest]::Create($t.url)
      $req.Timeout = 60000; $req.ReadWriteTimeout = 120000
      if ($have -gt 0) { $req.AddRange($have) }
      $resp = $req.GetResponse()
      $rs = $resp.GetResponseStream()
      $fs = New-Object IO.FileStream($t.out, [IO.FileMode]::Append, [IO.FileAccess]::Write)
      $buf = New-Object byte[] 1048576
      while (($n = $rs.Read($buf,0,$buf.Length)) -gt 0) { $fs.Write($buf,0,$n) }
      $fs.Close(); $rs.Close(); $resp.Close()
      Write-Output "SEGMENT_OK $($t.out) now=$((Get-Item $t.out).Length) attempt=$attempt"
    } catch {
      Write-Output "RETRY $attempt $($t.out) $($_.Exception.Message)"
      Start-Sleep -Seconds 5
    }
  }
}
Write-Output "DLDONE"
