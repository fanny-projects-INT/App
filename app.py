from pathlib import Path
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Test DB read", layout="wide")

DB_URL = "https://github.com/fanny-projects-INT/App/releases/latest/download/full_db_all_rigs.feather"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "full_db_all_rigs.feather"

st.title("Test download + read feather")

st.write("DB_URL:", DB_URL)
st.write("DB_PATH:", str(DB_PATH))

try:
    if not DB_PATH.exists():
        st.write("Downloading file...")
        with requests.get(DB_URL, stream=True, timeout=300) as response:
            response.raise_for_status()
            total = 0
            with open(DB_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        if total % (50 * 1024 * 1024) < 1024 * 1024:
                            st.write(f"Downloaded ~ {total / (1024 * 1024):.1f} MB")

    st.success(f"File present: {DB_PATH.exists()}")
    st.write("File size (MB):", round(DB_PATH.stat().st_size / (1024 * 1024), 2))

    st.write("Reading feather with pandas...")
    df = pd.read_feather(DB_PATH)

    st.success("Feather loaded successfully")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist()[:20])
    st.dataframe(df.head())

except Exception as e:
    st.error("Failure during download or read_feather")
    st.exception(e)