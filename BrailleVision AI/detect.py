# detect.py

import cv2
import numpy as np


def _odd(value):
    value = int(value)
    if value < 3:
        return 3
    return value if value % 2 == 1 else value + 1


def _normalize_lighting(gray):
    """Flatten shadows so physical Braille dots stand out from paper texture."""

    denoised = cv2.bilateralFilter(gray, 7, 45, 45)
    min_dim = min(gray.shape[:2])
    background_kernel = _odd(max(31, min_dim // 5))
    background = cv2.GaussianBlur(
        denoised,
        (background_kernel, background_kernel),
        0,
    )

    normalized = cv2.divide(denoised, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)


def _build_dot_mask(normalized, sensitivity, polarity):
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    percentile_width = 5.0 + (sensitivity * 14.0)

    if polarity == "bright":
        percentile_cutoff = np.percentile(normalized, 100.0 - percentile_width)
        otsu_cutoff, _ = cv2.threshold(
            normalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        cutoff = max(percentile_cutoff, otsu_cutoff - (sensitivity * 8.0))
        mask = np.zeros_like(normalized, dtype=np.uint8)
        mask[normalized >= cutoff] = 255
    else:
        percentile_cutoff = np.percentile(normalized, percentile_width)
        otsu_cutoff, _ = cv2.threshold(
            normalized,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        cutoff = min(percentile_cutoff, otsu_cutoff + (sensitivity * 8.0))
        mask = np.zeros_like(normalized, dtype=np.uint8)
        mask[normalized <= cutoff] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask, float(cutoff)


def _local_contrast(gray, contour, x, y, w, h, polarity):
    pad = max(6, int(max(w, h) * 1.5))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)

    roi = gray[y1:y2, x1:x2]
    contour_roi = contour.copy()
    contour_roi[:, 0, 0] -= x1
    contour_roi[:, 0, 1] -= y1

    dot_mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.drawContours(dot_mask, [contour_roi], -1, 255, -1)

    ring_kernel_size = _odd(max(5, int(max(w, h) * 2.0)))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (ring_kernel_size, ring_kernel_size),
    )
    outer_mask = cv2.dilate(dot_mask, ring_kernel, iterations=1)
    ring_mask = cv2.bitwise_and(outer_mask, cv2.bitwise_not(dot_mask))

    if np.count_nonzero(dot_mask) == 0 or np.count_nonzero(ring_mask) == 0:
        return 0.0

    dot_mean = cv2.mean(roi, mask=dot_mask)[0]
    background_median = float(np.median(roi[ring_mask > 0]))

    if polarity == "bright":
        return dot_mean - background_median

    return background_median - dot_mean


def _dedupe_candidates(candidates):
    kept = []

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        cx, cy = candidate["center"]
        radius = candidate["radius"]

        duplicate = False
        for other in kept:
            ox, oy = other["center"]
            min_distance = max(radius, other["radius"]) * 0.75
            if (cx - ox) ** 2 + (cy - oy) ** 2 <= min_distance**2:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return sorted(kept, key=lambda item: (item["center"][1], item["center"][0]))


def _detect_for_polarity(image, gray, normalized, sensitivity, polarity):
    mask, cutoff = _build_dot_mask(normalized, sensitivity, polarity)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    image_area = image.shape[0] * image.shape[1]
    min_area = max(10.0, image_area * 0.000004)
    max_area = max(120.0, image_area * 0.008)
    min_contrast = 32.0 - (float(np.clip(sensitivity, 0.0, 1.0)) * 16.0)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.55 or aspect_ratio > 1.85:
            continue

        contour_mask = np.zeros(mask[y : y + h, x : x + w].shape, dtype=np.uint8)
        contour_in_roi = contour.copy()
        contour_in_roi[:, 0, 0] -= x
        contour_in_roi[:, 0, 1] -= y
        cv2.drawContours(contour_mask, [contour_in_roi], -1, 255, -1)
        dot_pixels = cv2.bitwise_and(mask[y : y + h, x : x + w], contour_mask)
        fill_ratio = np.count_nonzero(dot_pixels) / max(1, np.count_nonzero(contour_mask))
        if fill_ratio < 0.48:
            continue

        extent = area / float(w * h)
        if extent < 0.32:
            continue

        contrast = _local_contrast(gray, contour, x, y, w, h, polarity)
        if contrast < min_contrast:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        candidates.append(
            {
                "center": (int(round(cx)), int(round(cy))),
                "radius": float(radius),
                "score": float(contrast * circularity),
                "contrast": float(contrast),
            }
        )

    candidates = _dedupe_candidates(candidates)

    return {
        "polarity": polarity,
        "mask": mask,
        "threshold_cutoff": cutoff,
        "raw_contours": len(contours),
        "candidates": candidates,
        "filtered_dots": len(candidates),
        "min_area": float(min_area),
        "max_area": float(max_area),
        "min_contrast": float(min_contrast),
    }


def _draw_candidates(image, candidates, polarity):
    output = image.copy()
    circle_color = (0, 170, 80) if polarity == "dark" else (20, 110, 255)

    for candidate in candidates:
        center = candidate["center"]
        radius = max(5, int(round(candidate["radius"] + 2)))
        cv2.circle(output, center, radius, circle_color, 2)
        cv2.circle(output, center, 2, (0, 80, 255), -1)

    return output


def detect_dots(image, sensitivity=0.45, polarity="auto", return_debug=False):
    """
    Detect physical Braille dot centers from a camera/photo image.

    polarity:
        auto   - choose dark or bright dot detection automatically
        dark   - dark printed/inked dots or shadow dots
        bright - embossed highlight dots
    """

    normalized_polarity = str(polarity).lower().strip()
    if normalized_polarity not in {"auto", "dark", "bright"}:
        normalized_polarity = "auto"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = _normalize_lighting(gray)

    if normalized_polarity == "auto":
        dark_result = _detect_for_polarity(image, gray, normalized, sensitivity, "dark")
        bright_result = _detect_for_polarity(image, gray, normalized, sensitivity, "bright")
        result = max(
            [dark_result, bright_result],
            key=lambda item: (
                item["filtered_dots"],
                sum(candidate["score"] for candidate in item["candidates"]),
            ),
        )
        result["auto_runs"] = {
            "dark_dots": dark_result["filtered_dots"],
            "bright_dots": bright_result["filtered_dots"],
        }
    else:
        result = _detect_for_polarity(
            image,
            gray,
            normalized,
            sensitivity,
            normalized_polarity,
        )
        result["auto_runs"] = None

    output = _draw_candidates(image, result["candidates"], result["polarity"])
    dots = [candidate["center"] for candidate in result["candidates"]]

    debug = {
        "mask": result["mask"],
        "normalized": normalized,
        "threshold_cutoff": result["threshold_cutoff"],
        "raw_contours": result["raw_contours"],
        "filtered_dots": result["filtered_dots"],
        "min_area": result["min_area"],
        "max_area": result["max_area"],
        "min_contrast": result["min_contrast"],
        "polarity": result["polarity"],
        "auto_runs": result["auto_runs"],
    }

    if return_debug:
        return output, dots, debug

    return output, dots
