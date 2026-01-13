import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch

# ──────────────────────────────────────────────────────────────
# Cache the model & processor (very important for Streamlit Cloud)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model... (first time takes ~1-2 min)")
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device


# ──────────────────────────────────────────────────────────────
# Main caption generation function
# ──────────────────────────────────────────────────────────────
def generate_caption(image, processor, model, device, conditional_text=""):
    inputs = processor(images=image, text=conditional_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
    
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption


# ──────────────────────────────────────────────────────────────
# Streamlit App UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image or paste a URL — get an automatic caption using **BLIP** (Salesforce) model from Hugging Face!")

# Load model once
processor, model, device = load_model()

# Tabs for different input methods
tab1, tab2 = st.tabs(["📤 Upload Image", "🌐 Image URL"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("✨ Generate Caption", type="primary"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image, processor, model, device)
                st.success("**Caption:** " + caption)

with tab2:
    url = st.text_input("Paste image URL here (must be direct link to .jpg/.png)")
    
    if url:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Image from URL", use_column_width=True)
            
            if st.button("✨ Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image, processor, model, device)
                    st.success("**Caption:** " + caption)
        except Exception as e:
            st.error(f"Could not load image from URL. Error: {str(e)}")

# Footer / info
st.markdown("---")
st.caption("Powered by Salesforce/blip-image-captioning-base • Runs on Streamlit Cloud • No API key needed")
