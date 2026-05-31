# sample_generator.py

from pathlib import Path

import cv2
import numpy as np

from braille_map import TEXT_TO_BRAILLE


ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "sample_images"


DOT_POSITIONS = {
    1: (0, 0),
    2: (0, 1),
    3: (0, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
}


def _paper_background(width, height, rng, difficulty):
    base = np.full((height, width, 3), 238, dtype=np.uint8)

    if difficulty == "beginner":
        return base

    noise_level = 5 if difficulty == "medium" else 10
    noise = rng.normal(0, noise_level, (height, width, 1)).astype(np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if difficulty == "high":
        gradient = np.linspace(1.0, 0.82, width, dtype=np.float32)
        gradient = np.tile(gradient, (height, 1))
        noisy = np.clip(noisy.astype(np.float32) * gradient[:, :, None], 0, 255).astype(np.uint8)

    return noisy


def _draw_dot(image, center, radius, difficulty, rng):
    if difficulty == "beginner":
        color = (22, 28, 36)
    elif difficulty == "medium":
        color = (36, 38, 45)
    else:
        shade = int(rng.integers(24, 54))
        color = (shade, shade, shade)

    cv2.circle(image, center, radius, color, -1, lineType=cv2.LINE_AA)

    if difficulty != "beginner":
        highlight = tuple(min(channel + 45, 255) for channel in color)
        cv2.circle(
            image,
            (center[0] - radius // 3, center[1] - radius // 3),
            max(2, radius // 4),
            highlight,
            -1,
            lineType=cv2.LINE_AA,
        )


def create_braille_sample(text, filename, difficulty):
    seed_text = f"{text}:{difficulty}"
    seed = sum((index + 1) * ord(char) for index, char in enumerate(seed_text))
    rng = np.random.default_rng(seed)

    dot_radius = {"beginner": 12, "medium": 11, "high": 10}[difficulty]
    row_gap = 44
    col_gap = 40
    cell_gap = 56
    space_gap = 78
    margin_x = 72
    margin_y = 72
    cell_pitch = col_gap + cell_gap

    visible_cells = sum(1 for char in text.upper() if char != " ")
    spaces = sum(1 for char in text.upper() if char == " ")
    width = margin_x * 2 + visible_cells * cell_pitch + spaces * space_gap
    height = margin_y * 2 + row_gap * 2 + dot_radius * 4

    image = _paper_background(width, height, rng, difficulty)

    x = margin_x
    y = margin_y + dot_radius

    for char in text.upper():
        if char == " ":
            x += space_gap
            continue

        pattern = TEXT_TO_BRAILLE.get(char)
        if pattern is None:
            x += cell_pitch
            continue

        for dot_number in pattern:
            col, row = DOT_POSITIONS[dot_number]
            jitter = 0 if difficulty == "beginner" else int(rng.integers(-2, 3))
            center = (
                int(x + col * col_gap + jitter),
                int(y + row * row_gap + jitter),
            )
            _draw_dot(image, center, dot_radius, difficulty, rng)

        x += cell_pitch

    if difficulty == "medium":
        image = cv2.GaussianBlur(image, (3, 3), 0)

    if difficulty == "high":
        image = cv2.GaussianBlur(image, (3, 3), 0)
        for _ in range(25):
            point = tuple(int(v) for v in rng.integers([0, 0], [width, height]))
            color = int(rng.integers(185, 225))
            cv2.circle(image, point, int(rng.integers(1, 3)), (color, color, color), -1)

    SAMPLE_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(SAMPLE_DIR / filename), image)


def main():
    create_braille_sample("ABC", "beginner_abc.png", "beginner")
    create_braille_sample("HELLO", "medium_hello.png", "medium")
    create_braille_sample("VISION AI", "high_vision_ai.png", "high")
    print("Sample images created in sample_images/")


if __name__ == "__main__":
    main()
