import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from gtts import gTTS
from PIL import Image
from io import BytesIO

# Image Captioning Model (BLIP)
def generate_image_caption(image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

# Translation Model (MarianMT)
def translate_text(text, src_lang="en", tgt_lang="ar"):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text

# Text-to-Speech (gTTS)
def text_to_speech(text, lang="ar"):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)  # إعادة المؤشر إلى بداية الملف
    return audio_bytes

# Streamlit layout
st.set_page_config(layout="wide")  

st.markdown("""
    <style>
    .stApp {
        background-color: #1F2937; 
    }
    h1 {
        color: #F3F4F6; 
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Segoe UI', sans-serif;
        font-weight: bold;
    }
    .css-1d391kg {
        background-color: #3B82F6; 
        color: #F3F4F6;
    }
    .stButton button {
        background-color: #EF4444;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.1rem;
        border: none;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: goldenrod; 
    }
    .stTextArea textarea {
        background-color: #F3F4F6;
        color: #1F2937;
        font-size: 1.1rem;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Segoe UI', sans-serif;
    }
    .stAudio {
        background-color: #111827;
        border-radius: 8px;
        padding: 12px;
    }
    h2, h3 {
        color: #60A5FA; 
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .css-1aumxhk {
        background-color: #111827;
        border: 2px solid #3B82F6;
        border-radius: 8px;
    }
    label {
        font-weight: bold;
        color: #F3F4F6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Image Captioning-Translate-TTS")

with st.sidebar:
    uploaded_image = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 2])

with col1:
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption='Uploaded Image', use_column_width=True)

# Display the results
with col2:
    st.subheader("Results")

    caption_placeholder = st.empty()  
    translated_placeholder = st.empty()  
    audio_placeholder = st.empty() 

    if uploaded_image is not None:
        # Generate image caption
        caption = generate_image_caption(img)
        caption_placeholder.text_area(" **Generated Caption**", value=caption, height=100)

        # Translate caption
        translated_caption = translate_text(caption)
        translated_placeholder.text_area(" **Translated Text (Arabic)**", value=translated_caption, height=100)

        # Convert translated text to speech and create audio
        audio_bytes = text_to_speech(translated_caption)
        audio_placeholder.audio(audio_bytes, format='audio/mp3')

        # Provide download button for the audio
        st.download_button(label=" Download Audio", data=audio_bytes, file_name="output_audio.mp3", mime="audio/mp3")
