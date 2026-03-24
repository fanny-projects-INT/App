from pathlib import Path
import requests
import duckdb
import streamlit as st
import pandas as pd

st.set_page_config(page_title="DuckDB test", layout="wide")

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


@st.cache_resource
def get_connection(db_path: str):
    return duckdb.connect(db_path, read_only=True)


@st.cache_data(show_spinner=False)
def get_mouse_list(db_path: str):
    con = duckdb.connect(db_path, read_only=True)
    mice = con.execute("""
        SELECT DISTINCT Mouse_ID
        FROM data
        ORDER BY Mouse_ID
    """).fetchall()
    con.close()
    return [m[0] for m in mice]


@st.cache_data(show_spinner=False)
def load_mouse_data(db_path: str, mouse_id: str):
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("""
        SELECT *
        FROM data
        WHERE Mouse_ID = ?
        ORDER BY Date
    """, [mouse_id]).df()
    con.close()
    return df


st.title("DuckDB minimal test")

try:
    with st.spinner("Downloading database..."):
        local_db_path = ensure_db_local()

    st.success("Database file available")
    st.write("DB path:", local_db_path)
    st.write("DB size (MB):", round(Path(local_db_path).stat().st_size / (1024 * 1024), 2))

    with st.spinner("Reading mouse list..."):
        mouse_options = get_mouse_list(local_db_path)

    st.success(f"{len(mouse_options)} mice found")

    selected_mouse = st.selectbox("Choose mouse", mouse_options)

    with st.spinner(f"Loading data for {selected_mouse}..."):
        df_mouse = load_mouse_data(local_db_path, selected_mouse)

    st.success("Mouse data loaded")
    st.write("Shape:", df_mouse.shape)
    st.write("Columns:", df_mouse.columns.tolist())

    st.subheader("Preview")
    st.dataframe(df_mouse.head())

    if "Date" in df_mouse.columns:
        try:
            df_mouse["Date"] = pd.to_datetime(df_mouse["Date"], errors="coerce")
            st.write("Min date:", df_mouse["Date"].min())
            st.write("Max date:", df_mouse["Date"].max())
        except Exception as e:
            st.warning(f"Date parsing warning: {e}")

except Exception as e:
    st.error("Test failed")
    st.exception(e)