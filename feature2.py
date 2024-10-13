import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import os
import torch
from transformers import pipeline as hf_pipeline
from groq import Groq

asr_model = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if torch.cuda.is_available() else -1)

def record_audio(duration, fs=44100, file_name="output.wav"):
    st.write("Recording... Please speak.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  
    write(file_name, fs, audio)
    st.success(f"Audio saved at {file_name}.")
    return file_name

def convert_audio_to_text(audio_file_path):
    st.write(f"Transcribing {audio_file_path}...")
    transcript = asr_model(audio_file_path,return_timestamps=True)["text"]
    client = Groq(api_key="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")
    trans = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are an expert at identifying speakers in a conversation transcript and can reason through the dialogue step by step."
        },
        {
            "role": "user",
            "content": (
                f"Please identify who said what in this transcript. "
                f"For each line of dialogue, use a step-by-step chain of thought to determine the speaker. "
                f"If the speaker is known from context, label them accordingly (e.g., 'Person A', 'Person B'). "
                f"If the speaker cannot be identified, label them as 'Speaker'. "
                f"Here is the transcript: [{transcript}]."
                f"Please output the result in the format: 'Speaker: [Line of dialogue]'."
            )
        }
    ],
    model="llama3-8b-8192",
    max_tokens=5000
)

    st.success("Transcription complete.")
    return trans.choices[0].message.content

def get_bot_response(context):
    full_context = "\n".join(context)
    detailed_prompt = (
        f"Please create a detailed memo and summary from the following transcript:\n\n"
        f"Transcript:\n{full_context}\n\n"
        f"Outline key points, conclusions, and recommendations"
        f"Also give your own point of few in the end that how things can be done in a better way"
    )
    
    client = Groq(api_key="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")
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in summarizing and creating detailed memos from meeting transcripts."
            },
            {
                "role": "user",
                "content": detailed_prompt,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=5000
    )
    
    return chat_completion.choices[0].message.content

st.title("Business Meeting Memo Generator")
st.markdown("### Record audio of your meeting, transcribe it, and generate detailed memos.")

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'memo_summary' not in st.session_state:
    st.session_state.memo_summary = ""
if 'uploaded_transcript' not in st.session_state:
    st.session_state.uploaded_transcript = ""
if 'uploaded_memo_summary' not in st.session_state:
    st.session_state.uploaded_memo_summary = ""

tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

with tab1:
    duration = st.number_input("Enter recording duration (in seconds):", min_value=1, max_value=60, value=10)

    if st.button("Record Audio"):
        audio_file_name = record_audio(duration)
        st.write(f"Audio recorded and saved as {audio_file_name}")

    if st.button("Transcribe Audio"):
        audio_file_path = "output.wav"
        
        if os.path.exists(audio_file_path):
            st.session_state.transcript = convert_audio_to_text(audio_file_path)
            st.write("### Full Transcript:")
            st.write(st.session_state.transcript)

    if st.button("Generate Memo and Summary"):
        if st.session_state.transcript:
            context = [st.session_state.transcript]
            st.session_state.memo_summary = get_bot_response(context)
            st.write("### Memo and Summary:")
            st.write(st.session_state.memo_summary)
        else:
            st.error("No transcript found. Please record and transcribe audio first.")

    if st.session_state.transcript:
        st.download_button(
            label="Download Transcript",
            data=st.session_state.transcript,
            file_name='transcript.txt',
            mime='text/plain'
        )

    if st.session_state.memo_summary:
        st.download_button(
            label="Download Memo",
            data=st.session_state.memo_summary,
            file_name='memo_summary.txt',
            mime='text/plain'
        )

with tab2:
    st.header("Upload Audio File for Summary")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Audio file uploaded successfully.")
        
        if st.button("Transcribe Uploaded Audio"):
            st.session_state.uploaded_transcript = convert_audio_to_text("uploaded_audio.wav")
            st.write("### Full Transcript from Uploaded Audio:")
            st.write(st.session_state.uploaded_transcript)

        if st.button("Generate Memo and Summary from Uploaded Audio"):
            if st.session_state.uploaded_transcript:
                context = [st.session_state.uploaded_transcript]
                st.session_state.uploaded_memo_summary = get_bot_response(context)
                st.write("### Memo and Summary from Uploaded Audio:")
                st.write(st.session_state.uploaded_memo_summary)
            else:
                st.error("No transcript found. Please upload and transcribe audio first.")

        if st.session_state.uploaded_transcript:
            st.download_button(
                label="Download Transcript from Uploaded Audio",
                data=st.session_state.uploaded_transcript,
                file_name='uploaded_transcript.txt',
                mime='text/plain'
            )

        if st.session_state.uploaded_memo_summary:
            st.download_button(
                label="Download Memo from Uploaded Audio",
                data=st.session_state.uploaded_memo_summary,
                file_name='uploaded_memo_summary.txt',
                mime='text/plain'
            )

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50; 
        color: white; 
        border: none; 
        border-radius: 12px; 
        padding: 15px 32px; 
        text-align: center; 
        text-decoration: none; 
        display: inline-block; 
        font-size: 16px; 
        margin: 4px 2px; 
        transition-duration: 0.4s; 
        cursor: pointer; 
    }
    .stButton>button:hover {
        background-color: white; 
        color: black; 
        border: 2px solid #4CAF50;
    }
    .stMarkdown {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
