# app.py
import streamlit as st
from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

st.set_page_config(page_title="Image Classifier", page_icon="🖼️")
st.title("🖼️ Minimalist Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load Hugging Face model and processor
    @st.cache_resource(show_spinner=False)
    def load_model():
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        return processor, model

    processor, model = load_model()

    # Prepare image for model
    inputs = processor(images=image, return_tensors="pt")
    
    # Make prediction
    with st.spinner("Classifying..."):
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

    st.success(f"Predicted Class: **{predicted_label}**")
