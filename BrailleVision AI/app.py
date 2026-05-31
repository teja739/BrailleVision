# app.py

import html
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from detect import detect_dots
from guidance import analyze_scan_quality
from translator import dots_to_text
from tts import speak


ROOT = Path(__file__).parent
SAMPLE_IMAGES = {
    "Beginner - ABC": ROOT / "sample_images" / "beginner_abc.png",
    "Medium - HELLO": ROOT / "sample_images" / "medium_hello.png",
    "High - VISION AI": ROOT / "sample_images" / "high_vision_ai.png",
}
POLARITY_OPTIONS = {
    "Auto physical dots": "auto",
    "Dark printed / shadow dots": "dark",
    "Light embossed highlights": "bright",
}


st.set_page_config(
    page_title="BrailleVision",
    page_icon=":eye:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
    :root {
        --page-top: #f0f7ff;
        --page-bottom: #fff8ed;
        --ink: #122033;
        --muted: #64748b;
        --panel: #ffffff;
        --line: #d9e4f2;
        --navy: #0f1b33;
        --cyan: #06b6d4;
        --teal: #0f766e;
        --green: #16a34a;
        --violet: #7c3aed;
        --amber: #f59e0b;
        --rose: #e11d48;
        --soft-cyan: #e6fbff;
        --soft-green: #ecfdf5;
        --soft-amber: #fff7dd;
        --soft-rose: #fff1f2;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(6, 182, 212, 0.18), transparent 32rem),
            radial-gradient(circle at top right, rgba(245, 158, 11, 0.16), transparent 30rem),
            linear-gradient(135deg, var(--page-top) 0%, #f7fbff 48%, var(--page-bottom) 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(247, 251, 255, 0.88);
        border-bottom: 1px solid rgba(217, 228, 242, 0.85);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, p, label, span {
        color: var(--ink);
        letter-spacing: 0;
    }

    .hero {
        background:
            linear-gradient(135deg, rgba(15, 27, 51, 0.96), rgba(22, 78, 99, 0.95)),
            linear-gradient(90deg, rgba(6, 182, 212, 0.45), rgba(245, 158, 11, 0.35));
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        padding: 1.55rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 40px rgba(15, 27, 51, 0.16);
    }

    .hero-kicker {
        color: #8ff7e8;
        font-size: 0.82rem;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(2.45rem, 5vw, 4.4rem);
        line-height: 1;
        margin: 0;
        font-weight: 950;
    }

    .hero p {
        color: #d6f3ff;
        max-width: 850px;
        margin: 0.75rem 0 0;
        font-size: 1.05rem;
        font-weight: 600;
    }

    .scan-shell {
        background: rgba(255, 255, 255, 0.86);
        border: 1px solid rgba(217, 228, 242, 0.95);
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(15, 27, 51, 0.08);
    }

    .scan-title {
        color: var(--ink);
        font-size: 1.45rem;
        font-weight: 950;
        margin-bottom: 0.2rem;
    }

    .scan-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        font-weight: 650;
        margin-bottom: 0.9rem;
    }

    .section-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 900;
        margin: 0.3rem 0 0.55rem;
    }

    .result-box {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 7px solid var(--teal);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 94px;
        box-shadow: 0 10px 24px rgba(15, 27, 51, 0.07);
    }

    .result-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .result-text {
        color: var(--ink);
        font-size: clamp(1.9rem, 3vw, 2.9rem);
        line-height: 1.1;
        font-weight: 950;
        word-break: break-word;
    }

    .status {
        border-radius: 8px;
        padding: 0.9rem 1rem;
        border: 1px solid var(--line);
        background: var(--panel);
        color: var(--ink);
        font-weight: 750;
    }

    .status.good {
        background: var(--soft-green);
        border-color: #a7e5c1;
        color: #166534;
    }

    .status.warn {
        background: var(--soft-amber);
        border-color: #efcf74;
        color: #8a5a00;
    }

    .status.bad {
        background: var(--soft-rose);
        border-color: #f2a9b9;
        color: #be123c;
    }

    .tip {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 5px solid var(--cyan);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.45rem 0;
        color: var(--muted);
        font-weight: 700;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stCameraInput"] section {
        background: #ffffff;
        border: 1px dashed #8fb6d8;
        border-radius: 8px;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        box-shadow: 0 8px 20px rgba(15, 27, 51, 0.055);
    }

    [data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-weight: 850;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-weight: 950;
    }

    .stButton > button {
        border-radius: 8px;
        min-height: 2.85rem;
        background: linear-gradient(135deg, var(--teal), var(--cyan));
        border: 1px solid rgba(15, 118, 110, 0.35);
        color: #ffffff;
        font-weight: 900;
    }

    .stButton > button:hover:not(:disabled) {
        background: linear-gradient(135deg, #0b5f58, #0891b2);
        border-color: rgba(8, 145, 178, 0.55);
        color: #ffffff;
    }

    .stButton > button:disabled {
        background: #d5dde8;
        border-color: #c5cfdd;
        color: #667085;
    }

    textarea {
        border-radius: 8px !important;
        border-color: var(--line) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_image_from_path(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def load_image_from_stream(file_obj):
    file_bytes = np.asarray(bytearray(file_obj.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def browser_speak(text):
    safe_text = json.dumps(text)
    components.html(
        f"""
        <script>
        const msg = new SpeechSynthesisUtterance({safe_text});
        msg.volume = 1.0;
        msg.rate = 0.82;
        msg.pitch = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
    )


def render_status(level, message):
    st.markdown(
        f'<div class="status {level}">{html.escape(message)}</div>',
        unsafe_allow_html=True,
    )


def run_detection(image, sensitivity, polarity):
    signature = inspect.signature(detect_dots)

    if "polarity" in signature.parameters:
        return detect_dots(
            image.copy(),
            sensitivity=sensitivity,
            polarity=polarity,
            return_debug=True,
        )

    processed_image, dots, debug = detect_dots(
        image.copy(),
        sensitivity=sensitivity,
        return_debug=True,
    )
    debug["polarity"] = "dark"
    debug["auto_runs"] = None
    return processed_image, dots, debug


st.markdown(
    """
<div class="hero">
    <div class="hero-kicker">BrailleVision Hackathon 2026</div>
    <h1>BrailleVision</h1>
    <p>Scan physical Braille from paper and convert it into readable English text and speech.</p>
</div>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="scan-shell">
    <div class="scan-title">Scan Source</div>
    <div class="scan-subtitle">Choose camera, upload, or demo sample in one place.</div>
</div>
""",
    unsafe_allow_html=True,
)

source_col, settings_col = st.columns([1.4, 1])

with source_col:
    input_mode = st.radio(
        "Choose source",
        ["Upload photo", "Demo samples", "Camera scan"],
        horizontal=True,
        index=0,
        key="scan_source_v2",
        help="Use Upload photo for a direct demo without browser camera permissions.",
    )

    captured_file = None
    uploaded_file = None
    selected_sample = None

    if input_mode == "Upload photo":
        uploaded_file = st.file_uploader(
            "Upload physical Braille photo",
            type=["jpg", "jpeg", "png"],
        )
    elif input_mode == "Demo samples":
        selected_sample = st.selectbox("Demo sample", list(SAMPLE_IMAGES.keys()))
    else:
        captured_file = st.camera_input("Take a Braille photo")

with settings_col:
    dot_style = st.radio(
        "Dot appearance",
        list(POLARITY_OPTIONS.keys()),
    )
    sensitivity_value = st.slider(
        "Dot sensitivity",
        min_value=1,
        max_value=10,
        value=5,
        help="Lower rejects texture. Higher catches faint dots.",
    )
    show_mask = st.toggle("Show processing mask", value=False)


image = None
source_name = ""

try:
    if input_mode == "Upload photo" and uploaded_file is not None:
        image = load_image_from_stream(uploaded_file)
        source_name = uploaded_file.name
    elif input_mode == "Demo samples" and selected_sample is not None:
        image = load_image_from_path(SAMPLE_IMAGES[selected_sample])
        source_name = selected_sample
    elif input_mode == "Camera scan" and captured_file is not None:
        image = load_image_from_stream(captured_file)
        source_name = "Camera snapshot"
except FileNotFoundError:
    st.error("Sample images are missing. Run: python sample_generator.py")
    st.stop()


if image is None:
    render_status("warn", "Upload a Braille photo or choose a demo sample to start. Camera is optional.")
    st.markdown('<div class="section-title">Demo Path For Judges</div>', unsafe_allow_html=True)
    demo_col_1, demo_col_2, demo_col_3 = st.columns(3)
    with demo_col_1:
        st.metric("Beginner", "ABC")
    with demo_col_2:
        st.metric("Medium", "HELLO")
    with demo_col_3:
        st.metric("High", "VISION AI")
    st.stop()


polarity = POLARITY_OPTIONS[dot_style]
processed_image, dots, debug = run_detection(
    image,
    sensitivity=sensitivity_value / 10.0,
    polarity=polarity,
)
translation = dots_to_text(dots)
recognized_text = translation["text"]
confidence_percent = int(round(translation["confidence"] * 100))
quality = analyze_scan_quality(image, dots, translation, debug)


metric_1, metric_2, metric_3, metric_4 = st.columns(4)
with metric_1:
    st.metric("Input", source_name)
with metric_2:
    st.metric("Detected dots", len(dots))
with metric_3:
    st.metric("Braille cells", len(translation["cells"]))
with metric_4:
    st.metric("Confidence", f"{confidence_percent}%")


render_status(quality["level"], quality["summary"])

st.divider()

image_col, overlay_col = st.columns(2)
with image_col:
    st.markdown('<div class="section-title">Physical Braille Input</div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

with overlay_col:
    st.markdown('<div class="section-title">Detected Dot Overlay</div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)


if show_mask:
    st.markdown('<div class="section-title">Processing Mask</div>', unsafe_allow_html=True)
    st.image(debug["mask"], use_container_width=True, clamp=True)


st.divider()

safe_result = html.escape(recognized_text if recognized_text else "No text found")
st.markdown(
    f"""
<div class="result-box">
    <div class="result-label">Recognized English Text</div>
    <div class="result-text">{safe_result}</div>
</div>
""",
    unsafe_allow_html=True,
)

editable_text = st.text_area(
    "Editable recognized text",
    value=recognized_text,
    height=95,
    label_visibility="collapsed",
)

action_col_1, action_col_2 = st.columns(2)
speech_text = f"BrailleVision recognized the physical Braille text as {editable_text}."

with action_col_1:
    if st.button("Speak recognized text", disabled=not editable_text.strip()):
        browser_speak(speech_text)
        success, error = speak(speech_text, rate=122, volume=1.0)

        if success:
            st.success("Speech started at full volume.")
        else:
            st.warning(f"Browser speech started. Local speaker fallback could not run: {error}")

with action_col_2:
    if st.button("Speak scan guidance"):
        browser_speak(quality["spoken"])
        success, error = speak(quality["spoken"], rate=128, volume=1.0)

        if success:
            st.success("Guidance spoken.")
        else:
            st.warning(f"Browser guidance started. Local speaker fallback could not run: {error}")


st.markdown('<div class="section-title">Camera Guidance</div>', unsafe_allow_html=True)
for tip in quality["tips"]:
    st.markdown(f'<div class="tip">{html.escape(tip)}</div>', unsafe_allow_html=True)


with st.expander("Technical detection details"):
    if translation["cells"]:
        st.dataframe(
            [
                {
                    "cell": index + 1,
                    "pattern": ",".join(str(dot) for dot in cell["pattern"]),
                    "letter": cell["letter"],
                }
                for index, cell in enumerate(translation["cells"])
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No cells detected.")

    st.write(f"Polarity used: {debug.get('polarity', 'dark')}")
    st.write(f"Raw contours: {debug['raw_contours']}")
    st.write(f"Filtered dots: {debug['filtered_dots']}")
    st.write(f"Brightness: {quality['brightness']:.1f}")
    st.write(f"Contrast: {quality['contrast']:.1f}")

    if debug.get("auto_runs"):
        st.write(f"Auto dark dots: {debug['auto_runs']['dark_dots']}")
        st.write(f"Auto bright dots: {debug['auto_runs']['bright_dots']}")
