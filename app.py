from pathlib import Path
import zipfile
import requests
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Behavior dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_URL = "https://github.com/fanny-projects-INT/App/releases/latest/download/app_cache.zip"

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CACHE_ZIP = DATA_DIR / "app_cache.zip"
CACHE_DIR = DATA_DIR / "app_cache"
META_PATH = CACHE_DIR / "metadata.parquet"

YELLOW = "#F6E06E"
BLUE = "#90C5FF"
RED = "#F2A093"
NAVY = "#223248"
CARD_BORDER = "#E3EAF2"
PAGE_BG = "#F7FAFC"
MUTED = "#748091"

PROTOCOL_HEX = {1: YELLOW, 2: BLUE, 3: RED}
PROTOCOL_LABELS = {1: "Training 1", 2: "Training 2", 3: "Task"}


# =============================================================================
# CSS
# =============================================================================
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {PAGE_BG};
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid {CARD_BORDER};
        }}

        .block-container {{
            max-width: 1480px;
            padding-top: 0.6rem;
            padding-bottom: 1.2rem;
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        .app-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.1rem;
        }}

        .app-subtitle {{
            color: {MUTED};
            margin-bottom: 0.6rem;
        }}

        /* 🔥 TITRES + LIGNES COMPACT */
        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.15rem;
        }}

        .soft-rule {{
            border: none;
            border-top: 1px solid {CARD_BORDER};
            margin: 0.25rem 0 0.6rem 0;
        }}

        .metric-card {{
            background: white;
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 14px;
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: {MUTED};
        }}

        .metric-value {{
            font-size: 1.2rem;
            font-weight: 700;
            color: {NAVY};
        }}

        .small-muted {{
            color: {MUTED};
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA
# =============================================================================
@st.cache_data(show_spinner=False)
def ensure_cache_local():
    if META_PATH.exists():
        return str(CACHE_DIR)

    with requests.get(CACHE_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(CACHE_ZIP, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    with zipfile.ZipFile(CACHE_ZIP, "r") as zf:
        zf.extractall(CACHE_DIR)

    return str(CACHE_DIR)


@st.cache_data(show_spinner=False)
def load_metadata(cache_dir: str):
    df = pd.read_parquet(Path(cache_dir) / "metadata.parquet")

    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    df["Date_norm"] = df.get("Date_norm", df["Date"]).dt.normalize()
    df["Mouse_ID"] = df.get("Mouse_ID", "").astype(str)
    df["Version"] = df.get("Version", "").astype(str)
    df["Protocol"] = pd.to_numeric(df.get("Protocol"), errors="coerce")

    return df.sort_values(["Mouse_ID", "Date"]).reset_index(drop=True)


def abs_cache_path(cache_dir, rel):
    if not rel:
        return None
    return Path(cache_dir) / rel


def show_image(path):
    if path and Path(path).exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info("Image not available.")


# =============================================================================
# UI HELPERS
# =============================================================================
def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="soft-rule">', unsafe_allow_html=True)


# =============================================================================
# APP
# =============================================================================
inject_css()

st.markdown('<div class="app-title">Behavior dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Precomputed behavioral analysis</div>', unsafe_allow_html=True)

try:
    cache_dir = ensure_cache_local()
    df = load_metadata(cache_dir)

    mouse_id = st.sidebar.selectbox(
        "Mouse",
        sorted(df["Mouse_ID"].unique())
    )

    df_mouse = df[df["Mouse_ID"] == mouse_id]

    # === TOP METRICS
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Mouse", mouse_id)
    with c2:
        metric_card("Sessions", len(df_mouse))
    with c3:
        metric_card("Last session", df_mouse["Date"].max().strftime("%Y-%m-%d"))

    tab1, tab2 = st.tabs(["Overview", "Session focus"])

    # =============================================================================
    # OVERVIEW
    # =============================================================================
    with tab1:

        row = df_mouse.iloc[0]

        section("Session types")
        show_image(abs_cache_path(cache_dir, row["protocol_strip_path"]))

        section("Training progression")
        col1, col2 = st.columns(2)
        with col1:
            show_image(abs_cache_path(cache_dir, row["bout_count_rewards_path"]))
        with col2:
            show_image(abs_cache_path(cache_dir, row["stacked_lick_counts_path"]))

        section("Failure distributions")
        col3, col4 = st.columns(2)
        with col3:
            show_image(abs_cache_path(cache_dir, row["histogram_kde_failures_path"]))
        with col4:
            show_image(abs_cache_path(cache_dir, row["kde_failures_by_session_path"]))

        section("Regression")
        show_image(abs_cache_path(cache_dir, row["regression_rewards_failures_and_slope_path"]))

    # =============================================================================
    # SESSION FOCUS
    # =============================================================================
    with tab2:

        df_mouse["label"] = (
            df_mouse["Date"].dt.strftime("%Y-%m-%d")
            + " - v"
            + df_mouse["Version"]
        )

        selected = st.selectbox("Session", df_mouse["label"])
        row = df_mouse[df_mouse["label"] == selected].iloc[0]

        section("Session metadata")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Date", row["Date"].strftime("%Y-%m-%d"))
        with c2:
            metric_card("Version", row["Version"])
        with c3:
            metric_card("Protocol", PROTOCOL_LABELS.get(int(row["Protocol"]), "-"))

        c4, c5, c6 = st.columns(3)
        with c4:
            metric_card("Probas", row["Probas"])
        with c5:
            metric_card("Bouts", row["Number of Bouts"])
        with c6:
            metric_card("Rewards", row["Number of Rewarded Licks"])

        section("Session plots")

        col1, col2 = st.columns(2)
        with col1:
            show_image(abs_cache_path(cache_dir, row["session_rewards_vs_failures_path"]))
        with col2:
            show_image(abs_cache_path(cache_dir, row["session_failure_distribution_path"]))

except Exception as e:
    st.error("App failed")
    st.exception(e)