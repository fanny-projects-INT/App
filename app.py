from pathlib import Path
import zipfile
import requests
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Behavior dashboard", layout="wide")

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

    return df.sort_values(["Mouse_ID", "Date", "Version"]).reset_index(drop=True)


def abs_cache_path(cache_dir: str, rel_path: str | None):
    if not rel_path:
        return None
    return Path(cache_dir) / Path(rel_path)


def image_exists(path):
    return path is not None and Path(path).exists()


def show_image_if_exists(path, caption=None):
    if image_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info("Image not available.")


st.title("Behavior dashboard")

try:
    with st.spinner("Loading precomputed cache..."):
        local_cache_dir = ensure_cache_local()
        df = load_metadata(local_cache_dir)

    st.success("Cache loaded")
    st.write("Sessions:", len(df))
    st.write("Mice found:", df["Mouse_ID"].nunique())

    mouse_options = sorted(df["Mouse_ID"].dropna().unique().tolist())
    mouse_id = st.selectbox("Choose mouse", mouse_options)

    df_mouse = df[df["Mouse_ID"] == mouse_id].copy()

    st.subheader("Session table")
    st.dataframe(
        df_mouse[[
            "Date",
            "Version",
            "Protocol",
            "Probas",
            "Number of Bouts",
            "Number of Rewarded Licks",
        ]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Overview")

    if not df_mouse.empty:
        first_row = df_mouse.iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            show_image_if_exists(abs_cache_path(local_cache_dir, first_row.get("protocol_strip_path")))
            show_image_if_exists(abs_cache_path(local_cache_dir, first_row.get("stacked_lick_counts_path")))
            show_image_if_exists(abs_cache_path(local_cache_dir, first_row.get("kde_failures_by_session_path")))
        with col2:
            show_image_if_exists(abs_cache_path(local_cache_dir, first_row.get("bout_count_rewards_path")))
            show_image_if_exists(abs_cache_path(local_cache_dir, first_row.get("histogram_kde_failures_path")))
            show_image_if_exists(abs_cache_path(local_cache_dir, first_row.get("regression_rewards_failures_and_slope_path")))

    st.subheader("Session focus")

    df_mouse["date_str"] = df_mouse["Date"].dt.strftime("%Y-%m-%d")
    df_mouse["session_label"] = (
        df_mouse["date_str"]
        + " - v"
        + df_mouse["Version"].astype(str)
        + " - "
        + df_mouse["Protocol"].fillna(-1).astype(int).map(PROTOCOL_LABELS).fillna("Protocol")
    )

    session_label = st.selectbox("Choose session", df_mouse["session_label"].tolist())
    row = df_mouse[df_mouse["session_label"] == session_label].iloc[0]

    st.markdown(
        f"""
        **Date:** {row['date_str']}  
        **Version:** {row['Version']}  
        **Protocol:** {PROTOCOL_LABELS.get(int(row['Protocol']), str(row['Protocol'])) if pd.notna(row['Protocol']) else '-'}  
        **Probas:** {row['Probas']}  
        **Number of Bouts:** {row['Number of Bouts']}  
        **Number of Rewarded Licks:** {row['Number of Rewarded Licks']}
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image_if_exists(
            abs_cache_path(local_cache_dir, row.get("session_rewards_vs_failures_path")),
            caption="Rewards vs failures",
        )
    with col2:
        show_image_if_exists(
            abs_cache_path(local_cache_dir, row.get("session_failure_distribution_path")),
            caption="Failure distribution",
        )

except Exception as e:
    st.error("App failed")
    st.exception(e)