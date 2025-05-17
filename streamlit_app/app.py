import requests, json, io
from pathlib import Path
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

FASTAPI_URL = "http://localhost:8000/predict"   # same machine
TOPK = 3

st.set_page_config(page_title="Plant Disease Detector",
                   page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detector")
st.markdown(
    "Upload a leaf photo and the model will predict the most likely disease "
    "(trained on the PlantDoc dataset).")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Predicting…"):
        resp = requests.post(
            FASTAPI_URL,
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=20,
        )
    if resp.status_code != 200:
        st.error(f"API error: {resp.text}")
    else:
        preds = resp.json()["predictions"]
        labels = [p["class"] for p in preds]
        probs  = [p["prob"]*100 for p in preds]

        st.subheader("Top predictions")
        for lbl, pr in zip(labels, probs):
            st.write(f"**{lbl}** – {pr:.2f} %")

        # bar chart
        fig, ax = plt.subplots()
        ax.barh(labels[::-1], probs[::-1])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)
