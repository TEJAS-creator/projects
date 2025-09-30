# app.py
import streamlit as st
import wikipedia
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="Wikipedia Explorer", page_icon="🌐")

st.title("🌐 Interactive Wikipedia Explorer")

# Input topic
topic = st.text_input("Enter a topic:")

if topic:
    try:
        # Get summary
        summary = wikipedia.summary(topic, sentences=3)
        st.subheader("Summary")
        st.write(summary)

        # Related topics / suggestions
        st.subheader("Related Topics")
        suggestions = wikipedia.search(topic)
        for s in suggestions[:5]:
            st.write(f"- {s}")

        # Images
        st.subheader("Images")
        page = wikipedia.page(topic)
        images = page.images
        for img_url in images[:3]:  # show only first 3 images
            try:
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_column_width=True)
            except:
                continue

        # More info button
        if st.button("More Info"):
            st.subheader("Full Content")
            st.write(page.content)

    except wikipedia.exceptions.DisambiguationError as e:
        st.error("Topic is ambiguous. Options: " + ", ".join(e.options[:5]))
    except wikipedia.exceptions.PageError:
        st.error("Topic not found.")
