# 🏓 AR Table Tennis

[![Python](https://img.shields.io/badge/python-3.8+-FFD43B?style=for-the-badge&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=mediapipe)](https://mediapipe.dev/)
[![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-GPL%20v3-F5A97F?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0)

**Real-time augmented reality table tennis with hand tracking**

> [Demo Video](https://youtu.be/_ov-N5kLVU8)

Play solo table tennis in your living room! Swing your hand to hit falling balls and score points before the 3-minute timer runs out.

---

## ✨ Features

- **Real-time hand tracking** using MediaPipe's state-of-the-art landmark detection
- **Physics-based swing detection** with velocity, angle, and palm orientation checks
- **Visual feedback** with green/yellow/red flash effects for hits, near-misses, and misses
- **Trajectory preview** showing your swing direction before the ball reaches the net
- **Tutorial mode** for first-time players with guided practice balls
- **GPLv3 licensed** - fully open source and community-driven

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Windows, macOS, or Linux

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/ar-table-tennis.git
cd ar-table-tennis
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the game**

```bash
python game.py
```

That's it! The game window should open and start capturing your webcam feed.

---

## 🎮 How to Play

### Controls

| Key | Action |
|-----|--------|
| `Enter` | Start countdown / accept license |
| `Esc` | Exit game at any time |
| `Space` | Pause/resume during active rounds |
| `A` | View license overlay (while playing) |
| `Q` | Skip tutorial early |
| `Up/Down` | Scroll license text |
| `W/S` | Scroll license text |

### Game Flow

1. **Setup**: Adjust ball speed (300-800 px/s) using the trackbar
2. **Countdown**: Press Enter to start the 3-2-1 countdown
3. **Play**: 3-minute round where balls fall from the top
4. **Scoring**: Hit balls with a swift palm swing while in the player zone

### Swing Mechanics

To successfully hit a ball, your swing must meet all these criteria:

- ✅ **Collision**: Your palm center must be within 87.6px of the ball
- ✅ **Velocity**: Hand speed > 200 px/s (swifty motion!)
- ✅ **Palm orientation**: Palm facing the camera
- ✅ **Swing angle**: 20°-70° from horizontal (forward cone)

### Scoring

- 🟢 **Green flash**: Ball hit AND landed in opponent's zone = 1 point!
- 🟡 **Yellow flash**: Ball hit but didn't reach opponent's zone
- 🔴 **Red flash**: Ball missed (fell past the table bottom)

---

## 🛠️ Development

### Project Structure

```
ar-table-tennis/
├── game.py              # Main game script (all game logic)
├── hand_tracking.py     # Hand tracking utilities
├── requirements.txt     # Python dependencies
├── LICENSE              # GPLv3 license text
├── assets/              # Original SVG assets (source files)
├── _ASSETS/             # Pre-rasterized PNG assets (runtime)
├── hand_landmarker.task # MediaPipe hand landmark model
```

### Asset System

The game uses a two-tier asset system:

- **`assets/`** (SVG): Source files for table, racket, and balls
- **`_ASSETS/`** (PNG): Pre-rasterized PNG files with alpha channels for fast rendering

**Important**: The runtime uses the PNG files in `_ASSETS/`. The SVG files are provided for reference and customization.

### Customization

- **Ball colors**: Modify `game.py` to change spawn logic or add new ball types
- **Swing sensitivity**: Adjust `SWING_VELOCITY_THRESHOLD` and `SWING_NORMAL_DOT_THRESHOLD` constants
- **Collision radius**: Modify the `30px` hand radius in `evaluate_all_balls()`
- **Tutorial speed**: Change `TUTORIAL_BALL_SPEED` in the `TutorialSystem` class

---

## 🐛 Troubleshooting

### "Error: could not open webcam"

- Ensure your webcam is connected and not in use by another application
- Try unplugging and replugging external webcam
- Restart your computer if the camera remains unresponsive

### "Error: hand landmarker model not found"

The game requires the MediaPipe hand landmark model. Download it from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place the `hand_landmarker.task` file in the project root directory.

### "Error: SVG asset not found"

The game looks for PNG files in the `_ASSETS/` folder. Ensure these files exist:
- `table-net.png`
- `racket-red.png`
- `racket-black.png`
- `ball-white.png`
- `ball-orange.png`

If missing, check that the `_ASSETS/` folder contains the pre-rasterized images.

### "Error: LICENSE file not found"

The GPLv3 license text is required. Ensure a `LICENSE` file exists in the project root.

### High CPU usage or slow performance

- Close other applications to free up system resources
- Try a different webcam (some webcams are more GPU-intensive)
- Reduce the capture resolution in `game.py` (lines ~1200-1202)

### Hand tracking not working

- Ensure good lighting conditions
- Position your hand clearly visible in the frame
- Make sure your palm is facing the camera during swings
- The model requires 0.5 confidence threshold - extreme angles may not be detected

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥4.5 | Camera capture, rendering, UI |
| `mediapipe` | ≥0.10 | Hand landmark detection |
| `numpy` | ≥1.21 | Vector math and physics calculations |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code passes existing tests and follows the project's coding style.

---

## 📄 License

Distributed under the GPL v3 License. See `LICENSE` for details.

**AR Table Tennis** is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the excellent hand tracking solution
- [OpenCV](https://opencv.org/) for the powerful computer vision library
- [United Hacks V7](https://unitedhacksv7.devpost.com/) hackathon for the opportunity to create this project

---

**Happy playing!** 🏓✨
