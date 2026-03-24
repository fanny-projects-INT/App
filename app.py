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


PROTOCOL_HEX = {
    1: YELLOW,
    2: BLUE,
    3: RED,
}

PROTOCOL_LABELS = {
    1: "Training 1",
    2: "Training 2",
    3: "Task",
}


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
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        .app-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {NAVY};
            margin-top: 0;
            margin-bottom: 0.2rem;
            line-height: 1.15;
        }}

        .app-subtitle {{
            color: {MUTED};
            margin-bottom: 1rem;
        }}

        .soft-rule {{
            border: none;
            border-top: 1px solid {CARD_BORDER};
            margin: 0.6rem 0 1rem 0;
        }}

        .metric-card {{
            background: white;
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(34,50,72,0.04);
        }}

        .metric-label {{
            font-size: 0.88rem;
            color: {MUTED};
            margin-bottom: 0.18rem;
        }}

        .metric-value {{
            font-size: 1.2rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.2;
        }}

        .section-title {{
            font-size: 1.06rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.35rem;
        }}

        .small-muted {{
            color: {MUTED};
            font-size: 0.92rem;
        }}

        div[data-testid="stTabs"] button {{
            font-weight: 600;
        }}

        .stDataFrame {{
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    path = Path(cache_dir) / "metadata.parquet"
    df = pd.read_parquet(path).copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.NaT

    if "Date_norm" in df.columns:
        df["Date_norm"] = pd.to_datetime(df["Date_norm"], errors="coerce")
    else:
        df["Date_norm"] = df["Date"].dt.normalize()

    if "Mouse_ID" in df.columns:
        df["Mouse_ID"] = df["Mouse_ID"].astype(str)
    else:
        df["Mouse_ID"] = ""

    if "Version" in df.columns:
        df["Version"] = df["Version"].astype(str)
    else:
        df["Version"] = ""

    if "Protocol" in df.columns:
        df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")
    else:
        df["Protocol"] = pd.NA

    return df.sort_values(["Mouse_ID", "Date", "Version"]).reset_index(drop=True)


def abs_cache_path(cache_dir: str, rel_path: str | None):
    if not rel_path:
        return None
    return Path(cache_dir) / Path(rel_path)


def image_exists(path):
    return path is not None and Path(path).exists()


def show_image(path):
    if image_exists(path):
        st.image(str(path), use_container_width=True)
    else:
        st.info("Image not available.")


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


def section_header(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="small-muted">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="soft-rule">', unsafe_allow_html=True)


inject_css()

st.markdown('<div class="app-title">Behavior dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Precomputed overview and session-focus plots</div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading precomputed cache..."):
        local_cache_dir = ensure_cache_local()
        df = load_metadata(local_cache_dir)

    if df.empty:
        st.warning("No metadata found in cache.")
        st.stop()

    mouse_options = sorted(df["Mouse_ID"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown("## Navigation")
        mouse_id = st.selectbox("Mouse", mouse_options)
        st.markdown("---")
        st.caption(f"Sessions: {len(df)}")
        st.caption(f"Mice: {df['Mouse_ID'].nunique()}")

    df_mouse = df[df["Mouse_ID"] == mouse_id].copy()

    top1, top2, top3 = st.columns(3)
    with top1:
        metric_card("Mouse", mouse_id)
    with top2:
        metric_card("Sessions", len(df_mouse))
    with top3:
        latest_date = df_mouse["Date"].max()
        metric_card("Latest session", latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else "-")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Overview", "Session focus"])

    with tab1:
        if df_mouse.empty:
            st.info("No sessions for this mouse.")
        else:
            first_row = df_mouse.iloc[0]

            section_header("Session table")
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
            st.dataframe(
                df_mouse[show_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            section_header("Session types")
            show_image(abs_cache_path(local_cache_dir, first_row.get("protocol_strip_path")))

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            section_header("Training progression")
            c1, c2 = st.columns(2)
            with c1:
                show_image(abs_cache_path(local_cache_dir, first_row.get("bout_count_rewards_path")))
            with c2:
                show_image(abs_cache_path(local_cache_dir, first_row.get("stacked_lick_counts_path")))

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            section_header("Failure distributions")
            c3, c4 = st.columns(2)
            with c3:
                show_image(abs_cache_path(local_cache_dir, first_row.get("histogram_kde_failures_path")))
            with c4:
                show_image(abs_cache_path(local_cache_dir, first_row.get("kde_failures_by_session_path")))

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            section_header("Reward / failure regression")
            show_image(abs_cache_path(local_cache_dir, first_row.get("regression_rewards_failures_and_slope_path")))

    with tab2:
        if df_mouse.empty:
            st.info("No session available.")
        else:
            df_mouse = df_mouse.copy()
            df_mouse["date_str"] = df_mouse["Date"].dt.strftime("%Y-%m-%d")
            df_mouse["protocol_label"] = (
                df_mouse["Protocol"]
                .apply(lambda x: PROTOCOL_LABELS.get(int(x), f"Protocol {int(x)}") if pd.notna(x) else "-")
            )
            df_mouse["session_label"] = (
                df_mouse["date_str"]
                + " - v"
                + df_mouse["Version"].astype(str)
                + " - "
                + df_mouse["protocol_label"]
            )

            default_idx = len(df_mouse) - 1 if len(df_mouse) > 0 else 0
            session_label = st.selectbox(
                "Choose session",
                df_mouse["session_label"].tolist(),
                index=default_idx,
            )

            row = df_mouse[df_mouse["session_label"] == session_label].iloc[0]

            section_header("Session metadata")
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Date", row["date_str"])
            with m2:
                metric_card("Version", row["Version"])
            with m3:
                metric_card("Protocol", row["protocol_label"])

            m4, m5, m6 = st.columns(3)
            with m4:
                metric_card("Probas", row["Probas"])
            with m5:
                metric_card("Number of Bouts", row["Number of Bouts"])
            with m6:
                metric_card("Rewarded Licks", row["Number of Rewarded Licks"])

            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

            section_header("Session plots")
            col1, col2 = st.columns(2)
            with col1:
                show_image(abs_cache_path(local_cache_dir, row.get("session_rewards_vs_failures_path")))
            with col2:
                show_image(abs_cache_path(local_cache_dir, row.get("session_failure_distribution_path")))

except Exception as e:
    st.error("App failed")
    st.exception(e)