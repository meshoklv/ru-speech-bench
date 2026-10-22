# Contributing

- **Add an engine:** copy a card in `docs/engines/`, fill in the license with a link and the date you read it, the hardware, the exact command
  and the measured numbers. Say "not measured" instead of guessing.
- **Add numbers from other hardware:** open a PR that adds a row or a paragraph to the card with the hardware named.
- **Scripts:** no hard-coded paths, take paths from CLI arguments or environment variables, keep `python -m compileall stt tts vad bench` green.
- Do not add audio you do not own, model weights or reference voices of real people.
