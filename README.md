<div align="center">
<h2>
    MeloPlus: Advanced Python Library for MeloTts
</h2>
<div>
    <img width="500" alt="teaser" src="doc/logo.png">
</div>
<div>
    <a href="https://pypi.org/project/meloplus" target="_blank">
        <img src="https://img.shields.io/pypi/pyversions/meloplus.svg?color=%2334D058" alt="Supported Python versions">
    </a>
    <a href="https://badge.fury.io/py/meloplus"><img src="https://badge.fury.io/py/meloplus.svg" alt="pypi version"></a>
</div>
</div>

## 🛠️ Installation

```bash
pip install meloplus
```

## 🎙️ Usage

```python
from meloplus import MeloInference

# Speed is adjustable
speed = 1.0

# English
text = "Did you ever hear a folk tale about a giant turtle?"
model = MeloInference(language="EN", device="auto")
speaker_ids = model.hps.data.spk2id

# American accent
output_path = "en-us.wav"
model.tts_to_file(text, speaker_ids["EN-US"], output_path, speed=speed)
```

## 😍 Contributing

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## 📜 License

This project is licensed under the terms of the MIT License.

## 🤗 Citation

```bibtex
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```
