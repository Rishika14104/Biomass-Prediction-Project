import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------------
# Configuration
# -----------------------------

TARGET_COLUMNS = [
    "fresh_grass",
    "dry_grass",
    "fresh_white_clover",
    "dry_white_clover",
    "fresh_red_clover",
    "dry_red_clover",
    "fresh_clover",
    "dry_clover",
    "fresh_weeds",
    "dry_weeds",
    "dry_total"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, len(TARGET_COLUMNS))

    model.load_state_dict(
        torch.load(
            "biomass_resnet50.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model


model = load_model()

# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Image to Biomass Prediction",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Image to Biomass Prediction")

st.write(
"""
Upload a grass/clover field image and predict biomass values using
a ResNet50 Deep Learning model.
"""
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = transform(image)

    img = img.unsqueeze(0)

    img = img.to(device)

    if st.button("Predict Biomass"):

        with torch.no_grad():

            prediction = model(img)

            prediction = prediction.cpu().numpy()[0]

        st.success("Prediction Completed!")

        st.subheader("Predicted Biomass")

        for name, value in zip(TARGET_COLUMNS, prediction):

            st.write(f"**{name} : {value:.2f} g**")