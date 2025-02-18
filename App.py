import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load the pre-trained model and tokenizer for English to Arabic translation
model_name = 'Helsinki-NLP/opus-mt-en-ar'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate English to Arabic
def translate_english_to_arabic(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    translated_tokens = model.generate(**tokens)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Custom CSS for styling
st.markdown("""
    <style>
        /* Background color */
        body {
            background-color: #f5f5f5;
        }
        /* Title color and font */
        .stMarkdown h1 {
            color: #2c3e50;
            font-family: 'Arial', sans-serif;
        }
        /* Text input area */
        .stTextArea {
            background-color: #ffffff;
            color: #333333;
            border: 1px solid #ced4da;
            border-radius: 0.25rem;
        }
        /* Translate button */
        div.stButton > button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #2980b9;
        }
        /* Output text */
        .translated-text {
            background-color: #eaf2f8;
            color: #1c2833;
            padding: 1rem;
            border-radius: 0.25rem;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit App Interface
st.title('🌍 English to Arabic Translator')

st.markdown('### Enter the text you want to translate:')
english_text = st.text_area('', '')

if st.button('Translate'):
    if english_text:
        arabic_translation = translate_english_to_arabic(english_text)
        st.markdown('### Arabic Translation:')
        st.markdown(f"<div class='translated-text'>{arabic_translation}</div>", unsafe_allow_html=True)
    else:
        st.warning('Please enter text to translate.')
