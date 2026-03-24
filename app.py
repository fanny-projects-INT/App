import streamlit as st
import sys

st.set_page_config(page_title="Test App", layout="wide")

st.title("✅ App minimale OK")

st.write("Si tu vois ça, le déploiement Streamlit fonctionne.")

st.subheader("Infos debug")

st.write("Python version:", sys.version)

st.write("Streamlit fonctionne correctement 🎉")