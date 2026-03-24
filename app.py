from pathlib import Path
import requests
import duckdb
import streamlit as st
import pandas as pd

st.set_page_config(page_title="DuckDB light test", layout="wide")

DB_URL = "https://github.com/fanny-projects-INT/App/releases/latest/download/database.duckdb"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "database.duckdb"


@st.cache_data(show_spinner=False)
def ensure_db_local():
    if DB_PATH.exists():
        return str(DB_PATH)

    with requests.get(DB_URL, stream=True, timeout=300) as response:
        response.raise_for_status()
        with open(DB_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return str(DB_PATH)


def connect_db(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    return con


@st.cache_data(show_spinner=False)
def get_mouse_list(db_path: str):
    con = connect_db(db_path)
    rows = con.execute("""
        SELECT DISTINCT Mouse_ID
        FROM data
        ORDER BY Mouse_ID
    """).fetchall()
    con.close()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False)
def load_mouse_light(db_path: str, mouse_id: str):
    con = connect_db(db_path)

    # Colonnes "légères" seulement
    df = con.execute("""
        SELECT
            Mouse_ID,
            Date,
            Version,
            Protocol,
            Probas,
            "Number of Bouts",
            "Number of Rewarded Licks",
            Rewards,
            "Licks After",
            "Correct Bouts"
        FROM data
        WHERE Mouse_ID = ?
        ORDER BY Date
    """, [mouse_id]).df()

    con.close()
    return df


@st.cache_data(show_spinner=False)
def load_one_session_heavy(db_path: str, mouse_id: str, date_str: str, version: str = "1"):
    con = connect_db(db_path)

    # Colonnes lourdes seulement pour UNE session
    df = con.execute("""
        SELECT
            Mouse_ID,
            Date,
            Version,
            Protocol,
            Probas,
            Rewards,
            "Licks After",
            "Correct Bouts",
            Timestamps,
            "Bout for Timestamps",
            "Times Rewarded Licks",
            "Times Non Rewarded Licks",
            "Times Invalid Licks"
        FROM data
        WHERE Mouse_ID = ?
          AND CAST(Date AS DATE) = CAST(? AS DATE)
          AND Version = ?
        LIMIT 1
    """, [mouse_id, date_str, version]).df()

    con.close()
    return df


st.title("DuckDB light test")

try:
    with st.spinner("Downloading DB..."):
        local_db_path = ensure_db_local()

    st.success("DB available")
    st.write("Size (MB):", round(Path(local_db_path).stat().st_size / (1024 * 1024), 2))

    mouse_options = get_mouse_list(local_db_path)
    st.write("Mice found:", len(mouse_options))

    mouse_id = st.selectbox("Choose mouse", mouse_options)

    with st.spinner("Loading light mouse data..."):
        df_light = load_mouse_light(local_db_path, mouse_id)

    st.success("Light mouse data loaded")
    st.write("Shape:", df_light.shape)
    st.dataframe(df_light.head())

    if "Date" in df_light.columns:
        df_light["Date"] = pd.to_datetime(df_light["Date"], errors="coerce")
        valid_dates = sorted(df_light["Date"].dt.date.dropna().astype(str).unique().tolist())
    else:
        valid_dates = []

    if valid_dates:
        selected_date = st.selectbox("Choose session date", valid_dates)

        with st.spinner("Loading one heavy session..."):
            df_session = load_one_session_heavy(local_db_path, mouse_id, selected_date, "1")

        st.success("Heavy session loaded")
        st.write("Shape:", df_session.shape)
        st.dataframe(df_session.head())

except Exception as e:
    st.error("Test failed")
    st.exception(e)