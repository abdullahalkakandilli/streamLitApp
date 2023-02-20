import streamlit as st
from transformers import pipeline

from huggingface_hub import HfApi
from huggingface_hub import ModelFilter

st.set_page_config(page_title="Huggingface Course", page_icon="🤗")

@st.cache_resource
def load_hf_model(model: str):
    return pipeline("automatic-speech-recognition", model=model)

@st.cache_data
def fetch_asr_models():
    api = HfApi()
    whisper_models = api.list_models(filter=ModelFilter(
		task="automatic-speech-recognition",
        model_name="openai/",
	))
    wav2vec_models = api.list_models(filter=ModelFilter(
		task="automatic-speech-recognition",
        model_name="facebook/",
	))
    return [m.modelId for m in (whisper_models + wav2vec_models)]


st.title("Automatic Speech Recognition")

with st.sidebar:
    st.header("Configuration")
    all_models=sorted(fetch_asr_models())
    selected_model = st.selectbox(
        "Select model", 
        all_models,
        index=all_models.index("openai/whisper-large")
    )

pipeline = load_hf_model(selected_model)

with open("voiceover.wav", 'rb') as audio_file:
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

if st.button("Run Automatic Speech Recognition"):
    res = pipeline("voiceover.wav")
    st.write(res)