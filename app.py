import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

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
# Video Processor for capturing frames
# ──────────────────────────────────────────────────────────────
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        st.session_state['latest_frame'] = frame
        return frame


# ──────────────────────────────────────────────────────────────
# Streamlit App UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Image Caption Generator")
st.markdown("Upload an image, paste a URL, or use real-time webcam — get an automatic caption using **BLIP** (Salesforce) model from Hugging Face!")

# Load model once
processor, model, device = load_model()

# Tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🌐 Image URL", "📹 Real-Time Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        conditional_text = st.text_input("Optional prompt (e.g., 'a photo of')", key="upload_prompt")
        
        if st.button("✨ Generate Caption", type="primary", key="upload_btn"):
            with st.spinner("Generating caption..."):
                caption = generate_caption(image, processor, model, device, conditional_text)
                st.success("**Caption:** " + caption)

with tab2:
    url = st.text_input("Paste image URL here (must be direct link to .jpg/.png)")
    
    if url:
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Image from URL", use_column_width=True)
            
            conditional_text = st.text_input("Optional prompt (e.g., 'a photo of')", key="url_prompt")
            
            if st.button("✨ Generate Caption", type="primary", key="url_btn"):
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image, processor, model, device, conditional_text)
                    st.success("**Caption:** " + caption)
        except Exception as e:
            st.error(f"Could not load image from URL. Error: {str(e)}")

with tab3:
    st.header("Real-Time Webcam Captioning")
    st.markdown("Start your webcam below, then click 'Generate Caption' to caption the current frame.")
    
    if 'latest_frame' not in st.session_state:
        st.session_state['latest_frame'] = None
    
    webrtc_streamer(
        key="webcam",
        mode="sendrecv",
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
    )
    
    conditional_text = st.text_input("Optional prompt (e.g., 'a photo of')", key="webcam_prompt")
    
    if st.button("✨ Generate Caption", type="primary", key="webcam_btn"):
        if 'latest_frame' in st.session_state and st.session_state['latest_frame'] is not None:
            with st.spinner("Generating caption..."):
                frame = st.session_state['latest_frame']
                img_array = frame.to_ndarray(format="rgb24")
                image = Image.fromarray(img_array).convert("RGB")
                caption = generate_caption(image, processor, model, device, conditional_text)
                st.success("**Caption:** " + caption)
        else:
            st.error("No frame available. Ensure your webcam is running and allowed.")

# Footer / info
st.markdown("---")
st.caption("Powered by Salesforce/blip-image-captioning-base • Runs on Streamlit Cloud • No API key needed")
