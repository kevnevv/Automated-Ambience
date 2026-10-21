# 🎧 Automated-Ambience — AI-Powered Acoustic Mood DJ

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AI Model: YAMNet](https://img.shields.io/badge/Audio%20Classifier-Google%20YAMNet-orange.svg)](https://tfhub.dev/google/yamnet/1)
[![Backend: Flask](https://img.shields.io/badge/Backend-Flask%20%7C%20REST-green.svg)](https://flask.palletsprojects.com/)
[![Music: YouTube Music](https://img.shields.io/badge/Music%20Engine-yt--dlp%20%7C%20YTMusicAPI-red.svg)](https://github.com/yt-dlp/yt-dlp)

> **Automated-Ambience** is an autonomous ambient DJ that listens to acoustic signals in your room and automatically steers background audio to complement your real-time environment. Whether you are typing quietly, sharing dinner with friends, or focusing on work, the system dynamically curates and transitions between mood-aligned soundscapes without manual intervention.

---

## 🖥️ Live Player Interface

![Automated Ambience Player Interface](automated_ambience_homescreen.png)

*The adaptive glassmorphic interface dynamically shifts its theme color, ambient waveform, and status indicators based on real-time classification confidence and genre identity.*

---

## ⚡ How It Works

1. **Acoustic Classification (Google YAMNet)**: The system captures 1-second chunks of audio via `sounddevice` and classifies ambient noise across 521 acoustic event classes (keyboard typing, laughter, speech, silence).
2. **Weighted Threshold System**: Repeated environmental sounds accumulate points over time. Distinctive sounds (e.g. laughter, bird chirps) carry higher weight (1.5x) than continuous background noise (0.7x).
3. **Cross-Genre Breakthrough**: When a new mood is proposed, a 1.5x multiplier allows sustained environmental shifts to break through the currently playing mood cleanly.
4. **Dual-Deck Crossfade**: Two decoupled audio decks (`audioA` and `audioB`) pre-buffer incoming streams before executing a smooth 2-second equal-power crossfade to prevent audio pops.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install flask numpy sounddevice tensorflow tensorflow-hub ytmusicapi yt-dlp
```

### 2. Run the Server
```bash
python main.py
```

### 3. Open the Interface
* **Host URL**: `http://127.0.0.1:5000`
* **Network Remote**: `http://<your-local-ip>:5000` *(phone control/output)*
