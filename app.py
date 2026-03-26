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
DARK_GRAY = "#3A4758"

PROTOCOL_LABELS = {
    1: "Training 1",
    2: "Training 2",
    3: "Task",
}


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
            padding-top: 0.70rem;
            padding-bottom: 1.25rem;
        }}

        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        .sidebar-title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: {DARK_GRAY};
            line-height: 1.2;
            margin: 0 0 0.35rem 0;
            text-align: left;
        }}

        .page-title {{
            font-size: 1.95rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.15;
            margin: 0;
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

        .section-block {{
            margin-top: 8px;
            margin-bottom: 10px;
        }}

        .section-separator {{
            border: none;
            border-top: 1px solid {CARD_BORDER};
            margin: 0 0 8px 0;
        }}

        .section-title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: {NAVY};
            margin: 0;
            line-height: 1.2;
        }}

        .stDataFrame {{
            border: 1px solid {CARD_BORDER};
            border-radius: 14px;
            overflow: hidden;
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

    if "Valid Bouts" not in df.columns:
        if "Number of Bouts" in df.columns:
            df["Valid Bouts"] = df["Number of Bouts"]
        else:
            df["Valid Bouts"] = pd.NA

    return df.sort_values(["Mouse_ID", "Date", "Version"]).reset_index(drop=True)


def abs_cache_path(cache_dir, rel):
    if not rel:
        return None
    return Path(cache_dir) / rel


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


def bout_simple_card(valid_bouts, total_bouts):
    try:
        valid = int(valid_bouts) if pd.notna(valid_bouts) else 0
    except Exception:
        valid = 0

    try:
        total = int(total_bouts) if pd.notna(total_bouts) else 0
    except Exception:
        total = 0

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Valid Bouts</div>
            <div class="metric-value">{valid} / {total}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(
        f"""
        <div class="section-block">
            <hr class="section-separator">
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_card(path):
    with st.container(border=True):
        if path and Path(path).exists():
            st.image(str(path), use_container_width=True)
        else:
            st.caption("Image not available.")


def dataframe_card(dataframe):
    with st.container(border=True):
        st.dataframe(dataframe, use_container_width=True, hide_index=True)


# =============================================================================
# APP
# =============================================================================
inject_css()

try:
    cache_dir = ensure_cache_local()
    df = load_metadata(cache_dir)

    if df.empty:
        st.warning("No metadata found.")
        st.stop()

    latest_per_mouse = df.groupby("Mouse_ID", dropna=False)["Date"].max().dropna()
    if latest_per_mouse.empty:
        mouse_options = sorted(df["Mouse_ID"].unique().tolist())
        default_mouse = mouse_options[0]
    else:
        default_mouse = latest_per_mouse.idxmax()
        mouse_options = sorted(df["Mouse_ID"].unique().tolist())

    default_mouse_index = mouse_options.index(default_mouse) if default_mouse in mouse_options else 0

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Behavior dashboard</div>', unsafe_allow_html=True)

        mouse_id = st.selectbox(
            "Mouse",
            mouse_options,
            index=default_mouse_index,
        )

        df_mouse = df[df["Mouse_ID"] == mouse_id].copy()

        st.markdown("---")
        view_mode = st.radio("View", ["Overview", "Session focus"], index=0)
        st.markdown("---")
        st.caption(f"Sessions: {len(df_mouse)}")
        st.caption(f"Mice: {df['Mouse_ID'].nunique()}")

    st.markdown(
        f'<div class="page-title">{view_mode}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

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

        section("Session table")
        show_cols = [
            c for c in [
                "Date",
                "Version",
                "Protocol",
                "Probas",
                "Valid Bouts",
                "Number of Rewarded Licks",
            ] if c in df_mouse.columns
        ]
        dataframe_card(df_mouse[show_cols])

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        section("Plots")
        plot_card(abs_cache_path(cache_dir, row["protocol_strip_path"]))
        plot_card(abs_cache_path(cache_dir, row["bout_count_rewards_path"]))
        plot_card(abs_cache_path(cache_dir, row["stacked_lick_counts_path"]))

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            plot_card(abs_cache_path(cache_dir, row["histogram_kde_failures_path"]))
        with col2:
            plot_card(abs_cache_path(cache_dir, row["kde_failures_by_session_path"]))

        plot_card(abs_cache_path(cache_dir, row["regression_rewards_failures_and_slope_path"]))

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

        section("Session metadata")
        m1, m2 = st.columns(2, gap="medium")
        with m1:
            metric_card("Date", row["Date"].strftime("%Y-%m-%d"))
        with m2:
            metric_card(
                "Protocol",
                PROTOCOL_LABELS.get(int(row["Protocol"]), "-") if pd.notna(row["Protocol"]) else "-"
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        m3, m4 = st.columns(2, gap="medium")
        with m3:
            metric_card("Probas", row["Probas"])
        with m4:
            bout_simple_card(
                row.get("Valid Bouts"),
                row.get("Number of Bouts"),
            )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        section("Plots")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            plot_card(abs_cache_path(cache_dir, row["session_rewards_vs_failures_path"]))
        with col2:
            plot_card(abs_cache_path(cache_dir, row["session_failure_distribution_path"]))

except Exception as e:
    st.error("App failed")
    st.exception(e)