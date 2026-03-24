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
GRID = "#D9E1EA"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E8EEF5"
PAGE_BG = "#F7FAFC"

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
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            max-width: 1500px;
        }}

        .app-title {{
            font-size: 2rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.25rem;
        }}

        .app-subtitle {{
            color: #6E7B8C;
            margin-bottom: 1.2rem;
        }}

        .metric-card {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(34,50,72,0.04);
        }}

        .metric-label {{
            font-size: 0.9rem;
            color: #748091;
            margin-bottom: 0.2rem;
        }}

        .metric-value {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {NAVY};
        }}

        .section-card {{
            background: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 22px;
            padding: 18px 18px 12px 18px;
            margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(34,50,72,0.04);
        }}

        .section-title {{
            font-size: 1.08rem;
            font-weight: 700;
            color: {NAVY};
            margin-bottom: 0.75rem;
        }}

        .small-muted {{
            color: #748091;
            font-size: 0.92rem;
        }}

        div[data-testid="stTabs"] button {{
            font-weight: 600;
        }}

        .session-meta {{
            line-height: 1.8;
            color: {NAVY};
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


def show_image_card(path, title=None):
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if title:
            st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
        if image_exists(path):
            st.image(str(path), use_container_width=True)
        else:
            st.info("Image not available.")
        st.markdown("</div>", unsafe_allow_html=True)


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

        df_mouse_sidebar = df[df["Mouse_ID"] == mouse_id].copy()
        df_mouse_sidebar["date_str"] = df_mouse_sidebar["Date"].dt.strftime("%Y-%m-%d")
        df_mouse_sidebar["protocol_label"] = (
            df_mouse_sidebar["Protocol"]
            .apply(lambda x: PROTOCOL_LABELS.get(int(x), f"Protocol {int(x)}") if pd.notna(x) else "-")
        )

        if not df_mouse_sidebar.empty:
            df_mouse_sidebar["session_label"] = (
                df_mouse_sidebar["date_str"]
                + " - v"
                + df_mouse_sidebar["Version"].astype(str)
                + " - "
                + df_mouse_sidebar["protocol_label"]
            )
            default_idx = len(df_mouse_sidebar) - 1
            session_label = st.selectbox(
                "Session",
                df_mouse_sidebar["session_label"].tolist(),
                index=default_idx,
            )
        else:
            session_label = None

        st.markdown("---")
        st.markdown("## Cache")
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

    tab1, tab2 = st.tabs(["Overview", "Session focus"])

    with tab1:
        if df_mouse.empty:
            st.info("No sessions for this mouse.")
        else:
            first_row = df_mouse.iloc[0]

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Session table</div>', unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)

            row_a, row_b = st.columns(2)
            with row_a:
                show_image_card(
                    abs_cache_path(local_cache_dir, first_row.get("protocol_strip_path")),
                    "Session types",
                )
                show_image_card(
                    abs_cache_path(local_cache_dir, first_row.get("stacked_lick_counts_path")),
                    "Lick counts per session",
                )
                show_image_card(
                    abs_cache_path(local_cache_dir, first_row.get("kde_failures_by_session_path")),
                    "KDE of consecutive failures by task session",
                )

            with row_b:
                show_image_card(
                    abs_cache_path(local_cache_dir, first_row.get("bout_count_rewards_path")),
                    "Bout count and rewarded licks",
                )
                show_image_card(
                    abs_cache_path(local_cache_dir, first_row.get("histogram_kde_failures_path")),
                    "Distribution of consecutive failures",
                )
                show_image_card(
                    abs_cache_path(local_cache_dir, first_row.get("regression_rewards_failures_and_slope_path")),
                    "Reward/failure regression and slope",
                )

    with tab2:
        if df_mouse.empty or session_label is None:
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

            row = df_mouse[df_mouse["session_label"] == session_label].iloc[0]

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Session metadata</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="session-meta">
                    <b>Date:</b> {row['date_str']}<br>
                    <b>Version:</b> {row['Version']}<br>
                    <b>Protocol:</b> {row['protocol_label']}<br>
                    <b>Probas:</b> {row['Probas']}<br>
                    <b>Number of Bouts:</b> {row['Number of Bouts']}<br>
                    <b>Number of Rewarded Licks:</b> {row['Number of Rewarded Licks']}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                show_image_card(
                    abs_cache_path(local_cache_dir, row.get("session_rewards_vs_failures_path")),
                    "Rewards vs failures",
                )
            with col2:
                show_image_card(
                    abs_cache_path(local_cache_dir, row.get("session_failure_distribution_path")),
                    "Failure distribution",
                )

except Exception as e:
    st.error("App failed")
    st.exception(e)