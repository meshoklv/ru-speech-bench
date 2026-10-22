# IndexTTS-2

- **Kind:** TTS, local (GPU), zero-shot voice cloning
- **Upstream:** https://github.com/index-tts/index-tts
- **Status in this project:** **rejected without running** after reading the upstream README and source. No audio was synthesised and no numbers were measured.

## License of code and weights

**bilibili Model Use License Agreement** (title as in the repository `LICENSE`; the README states "This project is released under the bilibili Model Use License Agreement"). Checked 2026-10-22 on GitHub. It is a custom restricted-use license, not MIT/Apache. Key terms (summary, not legal advice):

- A worldwide, non-exclusive, non-transferable, royalty-free limited license (§2.1).
- Companies above 100 million monthly active users or RMB 1 billion annual revenue must request a separate license (§2.2).
- Outputs and derivatives may not be used to improve other AI models except the model itself, its derivatives or non-commercial models (§3.4(c)).
- Distributed derivatives need a non-endorsement disclaimer (§4.1(a)); some high-risk uses are restricted (§4.2).

Commercial use: **allowed for most users under these conditions, but not an open-source license; read it fully before use** (marked "conditional").

## Russian support (the reason for rejection)

An early brief claimed Russian was among nine supported languages and that the license was MIT. Neither held up:

- The README says the project "now supports **Chinese, English, Japanese, Spanish and Arabic**"; the word "Russian" does not appear anywhere in the roughly 25 KB README, and the example language parameter accepts only `ZH` and `EN`.
- Decisive evidence is in the code: the text front end (`indextts/utils/front.py`) defines exactly two normalizers, `zh_normalizer` and `en_normalizer`. There is no Russian text processing, so numbers, abbreviations and stress have nothing handling them.
- Checked on the repository as available during the original investigation (September/October 2026); the license text was re-read on 2026-10-22. The README language list was **not** re-read on 2026-10-22 -- re-check upstream if a newer release matters.

## Install and run

Not installed (about 5 GB of disk and time saved). Follow the upstream README if you want to test.

## Hardware

Planned test bench: NVIDIA RTX 3090 Ti, Windows 11. Nothing was run.

## Measured numbers

None.

## Known pitfalls

- Announcements and summaries about languages and licenses were wrong on both counts; read the repository README, `LICENSE` and the front-end code.

## Verdict

No Russian text front end and a custom, restricted license: not usable for Russian as released at the time of checking.
