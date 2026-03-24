import streamlit as st
import requests

st.set_page_config(page_title="Test DB URL", layout="wide")

DB_URL = "https://github.com/fanny-projects-INT/App/releases/latest/download/full_db_all_rigs.feather"

st.title("Test accès DB")

st.write("URL:", DB_URL)

try:
    response = requests.get(DB_URL, stream=True, timeout=60)
    st.write("Status code:", response.status_code)
    st.write("Content-Type:", response.headers.get("content-type"))
    st.write("Content-Length:", response.headers.get("content-length"))

    if response.status_code == 200:
        st.success("Le fichier est accessible depuis Streamlit Cloud.")
    else:
        st.error("Le fichier n'est pas accessible correctement.")

except Exception as e:
    st.exception(e)