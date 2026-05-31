# test_pipeline.py

from pathlib import Path

import cv2

from detect import detect_dots
from translator import dots_to_text


ROOT = Path(__file__).parent

SAMPLES = {
    "sample_images/beginner_abc.png": "ABC",
    "sample_images/medium_hello.png": "HELLO",
    "sample_images/high_vision_ai.png": "VISION AI",
}


def run_sample(path, expected):
    image = cv2.imread(str(ROOT / path))
    if image is None:
        raise FileNotFoundError(path)

    _, dots, debug = detect_dots(image, sensitivity=0.5, return_debug=True)
    result = dots_to_text(dots)
    actual = result["text"]

    print(f"{path}: expected={expected!r} actual={actual!r} dots={len(dots)}")

    if actual != expected:
        raise AssertionError(
            f"{path} failed. Expected {expected!r}, got {actual!r}. Debug: {debug}"
        )


def main():
    for path, expected in SAMPLES.items():
        run_sample(path, expected)

    print("All image-to-text sample tests passed.")


if __name__ == "__main__":
    main()
