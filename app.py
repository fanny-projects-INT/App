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
WHITE = "#FFFFFF"

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
            background: {WHITE};
            border-right: 1px solid {CARD_BORDER};
        }}

        .block-container {{
            max-width: 1460px;
            padding-top: 0.55rem;
            padding-bottom: 1.25rem;
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        .app-title {{
            font-size: 1.95rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.08rem;
            line-height: 1.15;
        }}

        .app-subtitle {{
            color: {MUTED};
            margin-bottom: 0.65rem;
        }}

        .section-wrap {{
            margin-top: 0.3rem;
            margin-bottom: 1rem;
        }}

        .section-rule {{
            border: none;
            border-top: 1px solid {CARD_BORDER};
            margin: 0 0 0.3rem 0;
        }}

        .section-title {{
            font-size: 1.03rem;
            font-weight: 700;
            color: {NAVY};
            margin: 0;
            line-height: 1.2;
        }}

        .metric-card {{
            background: {WHITE};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(34,50,72,0.04);
        }}

        .metric-label {{
            font-size: 0.84rem;
            color: {MUTED};
            margin-bottom: 0.16rem;
        }}

        .metric-value {{
            font-size: 1.18rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.2;
        }}

        .panel {{
            background: {WHITE};
            border: 1px solid {CARD_BORDER};
            border-radius: 18px;
            padding: 14px 14px 10px 14px;
            margin-bottom: 0.95rem;
            box-shadow: 0 1px 2px rgba(34,50,72,0.04);
        }}

        .small-muted {{
            color: {MUTED};
            font-size: 0.9rem;
        }}

        .stDataFrame {{
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            overflow: hidden;
        }}

        div[data-testid="stHorizontalBlock"] > div {{
            padding-right: 0.2rem;
            padding-left: 0.2rem;
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
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(CACHE_ZIP, "r") as zf:
        zf.extractall(CACHE_DIR)

    return str(CACHE_DIR)


@st.cache_data(show_spinner=False)
def load_metadata(cache_dir: str):
    df = pd.read_parquet(Path(cache_dir) / "metadata.parquet").copy()

    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    if "Date_norm" in df.columns:
        df["Date_norm"] = pd.to_datetime(df["Date_norm"], errors="coerce")
    else:
        df["Date_norm"] = df["Date"].dt.normalize()

    df["Mouse_ID"] = df.get("Mouse_ID", "").astype(str)
    df["Version"] = df.get("Version", "").astype(str)
    df["Protocol"] = pd.to_numeric(df.get("Protocol"), errors="coerce")

    return df.sort_values(["Mouse_ID", "Date", "Version"]).reset_index(drop=True)


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
    st.markdown(
        f"""
        <div class="section-wrap">
            <hr class="section-rule">
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_panel():
    st.markdown('<div class="panel">', unsafe_allow_html=True)


def close_panel():
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# APP
# =============================================================================
inject_css()

st.markdown('<div class="app-title">Behavior dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Precomputed behavioral analysis</div>', unsafe_allow_html=True)

try:
    cache_dir = ensure_cache_local()
    df = load_metadata(cache_dir)

    if df.empty:
        st.warning("No metadata found.")
        st.stop()

    mouse_id = st.sidebar.selectbox("Mouse", sorted(df["Mouse_ID"].unique().tolist()))

    df_mouse = df[df["Mouse_ID"] == mouse_id].copy()

    with st.sidebar:
        st.markdown("---")
        view_mode = st.radio("View", ["Overview", "Session focus"], index=0)
        st.markdown("---")
        st.caption(f"Sessions: {len(df_mouse)}")
        st.caption(f"Mice: {df['Mouse_ID'].nunique()}")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Mouse", mouse_id)
    with c2:
        metric_card("Sessions", len(df_mouse))
    with c3:
        latest_date = df_mouse["Date"].max()
        metric_card("Latest session", latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "-")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # OVERVIEW
    # =========================================================================
    if view_mode == "Overview":
        if df_mouse.empty:
            st.info("No sessions for this mouse.")
            st.stop()

        row = df_mouse.iloc[0]

        open_panel()
        section("Session table")
        show_cols = [
            c for c in [
                "Date",
                "Version",
                "Protocol",
                "Probas",
                "Number of Bouts",
                "Number of Rewarded Licks",
            ] if c in df_mouse.columns
        ]
        st.dataframe(df_mouse[show_cols], use_container_width=True, hide_index=True)
        close_panel()

        open_panel()
        section("Session types")
        show_image(abs_cache_path(cache_dir, row["protocol_strip_path"]))
        close_panel()

        open_panel()
        section("Training progression")
        show_image(abs_cache_path(cache_dir, row["bout_count_rewards_path"]))
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        show_image(abs_cache_path(cache_dir, row["stacked_lick_counts_path"]))
        close_panel()

        open_panel()
        section("Failure distributions")
        col1, col2 = st.columns(2)
        with col1:
            show_image(abs_cache_path(cache_dir, row["histogram_kde_failures_path"]))
        with col2:
            show_image(abs_cache_path(cache_dir, row["kde_failures_by_session_path"]))
        close_panel()

        open_panel()
        section("Regression")
        show_image(abs_cache_path(cache_dir, row["regression_rewards_failures_and_slope_path"]))
        close_panel()

    # =========================================================================
    # SESSION FOCUS
    # =========================================================================
    else:
        if df_mouse.empty:
            st.info("No session available.")
            st.stop()

        df_mouse = df_mouse.copy()
        df_mouse["label"] = (
            df_mouse["Date"].dt.strftime("%Y-%m-%d")
            + " - "
            + df_mouse["Protocol"].apply(
                lambda x: PROTOCOL_LABELS.get(int(x), f"Protocol {int(x)}") if pd.notna(x) else "-"
            )
        )

        default_idx = len(df_mouse) - 1 if len(df_mouse) > 0 else 0
        selected = st.selectbox("Session", df_mouse["label"].tolist(), index=default_idx)
        row = df_mouse[df_mouse["label"] == selected].iloc[0]

        open_panel()
        section("Session metadata")

        m1, m2 = st.columns(2, gap="small")
        with m1:
            metric_card("Date", row["Date"].strftime("%Y-%m-%d"))
        with m2:
            metric_card(
                "Protocol",
                PROTOCOL_LABELS.get(int(row["Protocol"]), "-") if pd.notna(row["Protocol"]) else "-"
            )

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        m3, m4 = st.columns(2, gap="small")
        with m3:
            metric_card("Probas", row["Probas"])
        with m4:
            metric_card("Number of Bouts", row["Number of Bouts"])
        close_panel()

        open_panel()
        section("Session plots")
        col1, col2 = st.columns(2)
        with col1:
            show_image(abs_cache_path(cache_dir, row["session_rewards_vs_failures_path"]))
        with col2:
            show_image(abs_cache_path(cache_dir, row["session_failure_distribution_path"]))
        close_panel()

except Exception as e:
    st.error("App failed")
    st.exception(e)