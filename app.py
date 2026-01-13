import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# 1. Page Configuration
st.set_page_config(page_title="AI Image Captioner", page_icon="📸")
st.title("📸 Real-time Image Captioning")

# 2. Load the AI Model (Cached for speed)
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# 3. Sidebar for Input Selection
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose Input Method:", ("Camera", "Upload File", "Image URL"))

image = None

# 4. Handle Inputs
if option == "Camera":
    # This opens the webcam or phone camera
    camera_photo = st.camera_input("Take a photo to caption")
    if camera_photo:
        image = Image.open(camera_photo).convert('RGB')

elif option == "Upload File":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

else:
    url = st.text_input("Paste Image URL:")
    if url:
        try:
            image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
        except:
            st.error("Invalid URL. Please provide a direct link to an image.")

# 5. Generate and Display Caption
if image:
    # Display the image
    st.image(image, caption="Current Image", use_container_width=True)
    
    with st.spinner('🤖 AI is describing your photo...'):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Display the result
        st.success(f"**AI Caption:** {caption.capitalize()}")
