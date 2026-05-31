# BrailleVision

BrailleVision is a hackathon-ready assistive technology demo for BrailleVision Hackathon 2026.

The project scans real physical Braille from a camera snapshot or uploaded photo, detects Braille dots with OpenCV, converts six-dot Braille cells into English text, and reads the result aloud.

This is not a Unicode Braille translator. The app processes camera/photo input of physical Braille dots from paper.

## Challenge Fit

- Real-world assistive technology for visually impaired users.
- Camera or photo input for physical Braille.
- Computer vision pipeline using OpenCV.
- English text output for caregivers, teachers, volunteers, and accessibility workers.
- Speech output for accessibility-focused demos.

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- pyttsx3

## Folder Structure

```text
BrailleVision/
|-- app.py
|-- braille_map.py
|-- detect.py
|-- guidance.py
|-- translator.py
|-- tts.py
|-- sample_generator.py
|-- test_pipeline.py
|-- requirements.txt
|-- README.md
|-- sample_images/
|   |-- beginner_abc.png
|   |-- medium_hello.png
|   `-- high_vision_ai.png
`-- outputs/
    `-- .gitkeep
```

## Main Features

- Camera snapshot mode for near-real-time physical Braille scanning.
- Upload mode for testing real Braille photos.
- Demo sample mode for reliable hackathon presentation.
- Automatic dark/bright dot detection for printed dots, shadow dots, and embossed highlights.
- Braille cell segmentation from dot coordinates.
- English A-Z translation.
- Full-volume speech output.
- Scan guidance for lighting, distance, clarity, and retake suggestions.
- Beginner, medium, and high difficulty sample images.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If the sample images are missing, generate them:

```bash
python sample_generator.py
```

## Run

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Demo Flow For Judges

1. Start the app.
2. Use `Demo samples`.
3. Select `Beginner - ABC`.
4. Show the detected dot overlay and recognized text.
5. Select `Medium - HELLO`.
6. Select `High - VISION AI`.
7. Click `Speak recognized text`.
8. Switch to `Camera scan` and take a physical Braille snapshot if camera access is available.
9. Use `Speak scan guidance` to show accessibility-focused feedback.

## Sample Input And Output

```text
sample_images/beginner_abc.png   -> ABC
sample_images/medium_hello.png   -> HELLO
sample_images/high_vision_ai.png -> VISION AI
```

## Testing

Run:

```bash
python test_pipeline.py
```

Expected output:

```text
sample_images/beginner_abc.png: expected='ABC' actual='ABC'
sample_images/medium_hello.png: expected='HELLO' actual='HELLO'
sample_images/high_vision_ai.png: expected='VISION AI' actual='VISION AI'
All image-to-text sample tests passed.
```

## Correct Camera Use

- Place the Braille paper on a flat surface.
- Use bright, even lighting.
- Avoid glare.
- Keep the camera straight above the paper.
- Crop close to the Braille dots.
- Start with `Auto physical dots`.
- Use `Dark printed / shadow dots` for inked or shadow-heavy dots.
- Use `Light embossed highlights` for bright raised dots.
- Adjust sensitivity only if dots are missed or paper texture is over-detected.

## Current Limitations

- Best results are on cropped, horizontal Braille.
- Supports A-Z letters and spaces for the hackathon demo.
- Numbers, punctuation, severe angle distortion, and long paragraphs need future work.
- True continuous video scanning can be added with Streamlit WebRTC or a mobile app camera API.

## Future Improvements

- Continuous webcam mode.
- Auto-crop Braille region.
- Skew correction for angled camera photos.
- Number and punctuation support.
- Mobile-first Android or Flutter version.
- Voice guidance such as move closer, move left, or improve lighting.
