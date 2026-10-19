
from pathlib import Path
from datetime import datetime
import io
import csv

import joblib
import librosa
import numpy as np
import soundfile as sf
import streamlit as st


# ==========================================
# ECOGUARD AI - PREMIUM DEMO WEBSITE
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

INSECT_MODEL_PATH = (
    BASE_DIR / "model" / "ecoguard_insect_model.joblib"
)
PLANT_MODEL_PATH = (
    BASE_DIR / "model" / "ecoguard_plant_model.joblib"
)

st.set_page_config(
    page_title="EcoGuard AI | Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PREMIUM GREEN DESIGN - WHITE HEADER FIX
# ==========================================

st.markdown("""
<style>
    /* Main page background */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(
            135deg,
            #071b13 0%,
            #0c281c 55%,
            #102e20 100%
        ) !important;
        color: #f0f7f1 !important;
    }

    /* FIX: Dark green top header */
    [data-testid="stHeader"] {
        background: #071b13 !important;
        color: #e4f7e9 !important;
        border-bottom: 1px solid #244735 !important;
    }

    /* FIX: Toolbar and top-right controls */
    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stHeader"] button {
        color: #e4f7e9 !important;
    }

    [data-testid="stHeader"] svg {
        fill: #e4f7e9 !important;
    }

    /* FIX: Top decoration strip */
    [data-testid="stDecoration"] {
        background: #071b13 !important;
    }

    /* Main content area */
    [data-testid="stMain"] {
        background: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #091c14 !important;
        border-right: 1px solid #244735 !important;
    }

    [data-testid="stSidebar"] * {
        color: #d0e5d5;
    }

    /* Headings */
    h1, h2, h3 {
        color: #e4f7e9 !important;
    }

    /* General text */
    p, label, li {
        color: #d0e5d5;
    }

    /* Dashboard metric cards */
    [data-testid="stMetric"] {
        background: #123524 !important;
        border: 1px solid #2d6144 !important;
        padding: 18px !important;
        border-radius: 14px !important;
    }

    [data-testid="stMetricLabel"] {
        color: #d0e5d5 !important;
    }

    [data-testid="stMetricValue"] {
        color: #8ee6a6 !important;
    }

    /* Buttons */
    div.stButton > button,
    div.stDownloadButton > button {
        background: #4fbd72 !important;
        color: #062013 !important;
        border: 1px solid #4fbd72 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 42px !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #79d994 !important;
        color: #062013 !important;
        border-color: #79d994 !important;
    }

    /* Audio recorder and file uploader */
    [data-testid="stFileUploader"],
    [data-testid="stAudioInput"] {
        background: #102c1e !important;
        border: 1px solid #315b41 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    /* Premium hero banner */
    .hero {
        background: linear-gradient(
            110deg,
            #153e29,
            #205d3b
        ) !important;
        border: 1px solid #3c7951;
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 22px;
    }

    .hero-title {
        color: #a6f0b7 !important;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #e1f4e5 !important;
        font-size: 16px;
    }

    /* Monitoring panels */
    .panel {
        background: #102b1e !important;
        border: 1px solid #2b553b;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .small-note {
        color: #a9c7b1 !important;
        font-size: 13px;
    }

    /* Dividers */
    hr {
        border-color: #2a4c36 !important;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #2b553b;
        border-radius: 12px;
    }

    /* Keep the app visually consistent */
    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD EXISTING SAVED MODELS
# ==========================================

@st.cache_resource
def load_models():
    if not INSECT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {INSECT_MODEL_PATH}"
        )

    if not PLANT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {PLANT_MODEL_PATH}"
        )

    insect_model = joblib.load(INSECT_MODEL_PATH)
    plant_model = joblib.load(PLANT_MODEL_PATH)

    return insect_model, plant_model


# ==========================================
# INSECT AUDIO FEATURES
# ==========================================

def extract_insect_features(audio_bytes):
    audio, sample_rate = librosa.load(
        io.BytesIO(audio_bytes),
        sr=22050,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("The recording is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "The recording contains invalid values."
        )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=20
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features.reshape(1, -1)


# ==========================================
# PLANT AUDIO FEATURES
# ==========================================

def extract_plant_features(audio_bytes):
    audio, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < 10:
        raise ValueError("The recording is too short.")

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "The recording contains invalid values."
        )

    if sample_rate <= 0:
        raise ValueError("Invalid sample rate.")

    mean_value = float(np.mean(audio))
    std_value = float(np.std(audio))
    rms_value = float(np.sqrt(np.mean(audio ** 2)))
    max_value = float(np.max(np.abs(audio)))

    centered = audio - np.mean(audio)

    spectrum = np.abs(np.fft.rfft(centered))

    frequencies = np.fft.rfftfreq(
        len(centered),
        d=1.0 / sample_rate
    )

    energy = spectrum ** 2
    total_energy = np.sum(energy) + 1e-12

    centroid = float(
        np.sum(frequencies * energy) / total_energy
    )

    spread = float(np.sqrt(
        np.sum(
            ((frequencies - centroid) ** 2) * energy
        ) / total_energy
    ))

    p25 = float(np.percentile(audio, 25))
    p75 = float(np.percentile(audio, 75))

    features = [
        mean_value,
        std_value,
        rms_value,
        max_value,
        centroid,
        spread,
        p25,
        p75
    ]

    return np.asarray(
        features,
        dtype=np.float32
    ).reshape(1, -1)


# ==========================================
# FARMER RECOMMENDATIONS
# ==========================================

def get_guidance(model_name, prediction):
    label = str(prediction).strip().lower()

    if model_name == "Insect & Pest Detection":

        if label == "pest":
            return (
                "Possible pest-related activity was flagged. "
                "Inspect leaves, stems, and nearby plants "
                "for visible evidence."
            ), True

        return (
            "The model selected the normal-audio category. "
            "This does not guarantee that pests are absent. "
            "Continue routine inspection."
        ), False

    if label == "water_stress":
        return (
            "Possible water stress was flagged. Check soil "
            "moisture, irrigation history, and plant condition "
            "before changing watering."
        ), True

    if label == "physical_damage":
        return (
            "Possible physical damage was flagged. Inspect "
            "the plant for broken stems, cuts, or other damage."
        ), True

    if label == "background":
        return (
            "The recording resembles the background category. "
            "This does not confirm that the plant is healthy."
        ), False

    return (
        "The model returned an unfamiliar category. "
        "Inspect the crop and verify the result."
    ), False


# ==========================================
# CREATE DOWNLOADABLE REPORT
# ==========================================

def create_report(record):
    actions = "\n".join(
        "- " + action for action in record["actions"]
    )

    return f"""
ECOGUARD AI - ANALYSIS REPORT
=============================

Date and time: {record['time']}
Model: {record['model']}
Recording: {record['file']}
Predicted category: {record['prediction']}
In-app alert: {"Yes" if record['alert'] else "No"}

Interpretation:
{record['guidance']}

Recommended actions:
{actions}

Limitations:
This is an experimental student prototype.
Predictions are not confirmed diagnoses.
SMS notifications are not configured.
"""


# ==========================================
# SESSION HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.markdown("## 🌱 EcoGuard AI")
st.sidebar.caption("SMART AGRICULTURE MONITORING")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "AI Sound Analysis",
        "Alert History",
        "About the Project"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.success("Website running")
st.sidebar.info("SMS: Not connected")


# ==========================================
# LOAD MODELS SAFELY
# ==========================================

try:
    insect_model, plant_model = load_models()

except Exception as error:
    st.error(f"Could not load the saved models: {error}")

    st.info(
        "Check that both .joblib files are inside "
        "the model folder."
    )

    st.stop()


# ==========================================
# HERO HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🌱 EcoGuard AI</div>
    <div class="hero-text">
        Early detection of crop stress and possible pest activity
        using experimental AI audio models.
    </div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Prototype warning: results may be incorrect. "
    "Verify every finding through crop inspection."
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    history = st.session_state.history

    total = len(history)

    pest_runs = sum(
        item["model"] == "Insect & Pest Detection"
        for item in history
    )

    plant_runs = sum(
        item["model"] == "Plant-Stress Detection"
        for item in history
    )

    alerts = sum(
        bool(item["alert"])
        for item in history
    )

    st.subheader("Monitoring Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total analyses", total)
    c2.metric("Pest model runs", pest_runs)
    c3.metric("Plant model runs", plant_runs)
    c4.metric("In-app alerts", alerts)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("""
        <div class="panel">
            <h3>🐛 Pest monitoring</h3>
            <p>Analyze audio using the insect model.</p>
            <p class="small-note">
                Categories: pest / normal
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="panel">
            <h3>🌿 Plant-stress monitoring</h3>
            <p>Analyze recordings using the plant model.</p>
            <p class="small-note">
                Categories: water stress / physical damage / background
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Recent Analyses")

    if history:
        for item in reversed(history[-5:]):
            st.write(
                f"**{item['time']}** · {item['model']} · "
                f"{item['prediction']} · {item['file']}"
            )
    else:
        st.info(
            "No analyses yet. Open AI Sound Analysis to begin."
        )

    st.caption(
        "This dashboard counts analyses in the current session only."
    )


# ==========================================
# AI SOUND ANALYSIS
# ==========================================

elif page == "AI Sound Analysis":

    st.subheader("🎙️ AI Sound Analysis")

    model_name = st.radio(
        "Choose the analysis model",
        [
            "Insect & Pest Detection",
            "Plant-Stress Detection"
        ],
        horizontal=True
    )

    st.markdown("### Choose an audio source")

    source = st.radio(
        "Audio input",
        [
            "Record with microphone",
            "Upload WAV file"
        ],
        horizontal=True
    )

    uploaded_audio = None

    if source == "Record with microphone":

        st.write(
            "Allow microphone access if your browser "
            "asks for permission."
        )

        if hasattr(st, "audio_input"):
            uploaded_audio = st.audio_input(
                "Record crop or environmental sound",
                key="ecoguard_microphone"
            )
        else:
            st.error(
                "Your Streamlit version does not support "
                "microphone recording. Run this command "
                "in the terminal: "
                "py -m pip install --upgrade streamlit"
            )

    else:
        uploaded_audio = st.file_uploader(
            "Choose a WAV recording",
            type=["wav"],
            key="ecoguard_upload"
        )

    if uploaded_audio is not None:

        audio_bytes = uploaded_audio.getvalue()

        st.audio(
            audio_bytes,
            format="audio/wav"
        )

        st.caption(
            f"Selected recording: {uploaded_audio.name}"
        )

        if st.button(
            "🌿 Analyze Sound",
            type="primary"
        ):

            try:
                with st.spinner(
                    "Analyzing sound with the selected AI model..."
                ):

                    if model_name == "Insect & Pest Detection":

                        features = extract_insect_features(
                            audio_bytes
                        )

                        if features.shape[1] != 40:
                            raise ValueError(
                                "Unexpected insect feature count."
                            )

                        prediction = insect_model.predict(
                            features
                        )[0]

                    else:
                        features = extract_plant_features(
                            audio_bytes
                        )

                        if features.shape[1] != 8:
                            raise ValueError(
                                "Unexpected plant feature count."
                            )

                        prediction = plant_model.predict(
                            features
                        )[0]

                guidance, alert = get_guidance(
                    model_name,
                    prediction
                )

                record = {
                    "time": datetime.now().astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    ),
                    "model": model_name,
                    "file": uploaded_audio.name,
                    "prediction": str(prediction),
                    "guidance": guidance,
                    "alert": alert,
                    "actions": [
                        "Inspect the crop to verify the result.",
                        "Record observations before taking action."
                    ]
                }

                st.session_state.history.append(record)

                st.divider()
                st.subheader("Analysis Results")

                a, b = st.columns(2)

                a.metric(
                    "Selected model",
                    model_name
                )

                b.metric(
                    "Predicted category",
                    str(prediction)
                )

                if alert:
                    st.warning(
                        "⚠️ Possible issue flagged by the model"
                    )
                else:
                    st.info("Model result recorded")

                st.markdown("### Farmer Recommendations")
                st.write(guidance)

                for action in record["actions"]:
                    st.write("• " + action)

                st.markdown("### Alert delivery")

                st.info(
                    "This result is displayed inside the website. "
                    "SMS and external notifications are not configured."
                )

                st.download_button(
                    "⬇️ Download Analysis Report",
                    data=create_report(record),
                    file_name="ecoguard_report.txt",
                    mime="text/plain"
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")

                st.caption(
                    "Check that the file is a valid WAV recording "
                    "and that you selected the appropriate model."
                )


# ==========================================
# ALERT HISTORY
# ==========================================

elif page == "Alert History":

    st.subheader("🔔 Analysis & Alert History")

    history = st.session_state.history

    if not history:
        st.info(
            "No analyses recorded in this session."
        )

    else:
        rows = []

        for item in reversed(history):
            rows.append({
                "Time": item["time"],
                "Model": item["model"],
                "File": item["file"],
                "Prediction": item["prediction"],
                "In-app alert": (
                    "Yes" if item["alert"] else "No"
                )
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )

        buffer = io.StringIO()

        writer = csv.DictWriter(
            buffer,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

        st.download_button(
            "⬇️ Download History CSV",
            data=buffer.getvalue(),
            file_name="ecoguard_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()


# ==========================================
# ABOUT THE PROJECT
# ==========================================

elif page == "About the Project":

    st.subheader("About EcoGuard AI")

    st.write(
        "EcoGuard AI is a student prototype exploring how "
        "audio classification could help monitor possible "
        "pest activity and plant stress."
    )

    st.markdown("### Current capabilities")

    st.markdown("""
    - Insect model: pest / normal audio classification.
    - Plant model: water stress / physical damage / background.
    - Microphone recording and WAV upload, when supported.
    - In-app results, recommendations, and session history.
    - Downloadable text reports and CSV history.
    """)

    st.markdown("### Current limitations")

    st.markdown("""
    - These models are experimental and may make mistakes.
    - The insect training dataset is very small.
    - The plant model has limited independent evaluation.
    - The plant model does not currently classify specific diseases.
    - Background or normal predictions do not guarantee crop health.
    - Session history is temporary.
    - SMS and external alerts are not connected.
    """)

    st.info(
        "For a real farm deployment, the models need broader "
        "field recordings and independent validation."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "EcoGuard AI | Smart Agriculture Student Prototype | "
    "Predictions require independent verification."
)
