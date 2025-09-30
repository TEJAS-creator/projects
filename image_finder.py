import streamlit as st
from PIL import Image
from streamlit_image_paste import image_paste
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

st.set_page_config(page_title="Image Classifier", page_icon="🖼️")
st.title("🖼️ Minimalist Image Classifier (Paste Image Supported)")

# Paste image directly
pasted_image = image_paste("Paste your image here (Ctrl+V)")

if pasted_image:
    # Convert to PIL image
    image = Image.open(pasted_image)
    st.image(image, caption="Pasted Image", use_column_width=True)

    # Load model (cached)
    @st.cache_resource(show_spinner=False)
    def load_model():
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        return processor, model

    processor, model = load_model()

    # Preprocess and classify
    inputs = processor(images=image, return_tensors="pt")
    with st.spinner("Classifying..."):
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

    st.success(f"Predicted Class: **{predicted_label}**")
