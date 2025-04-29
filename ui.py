import streamlit as st

import grpc

from generated import  massanger_pb2
from generated import massanger_pb2_grpc

# Setup gRPC connection
channel = grpc.insecure_channel("server:50051")
stub = massanger_pb2_grpc.TTSStub(channel)

st.title("🗣️ Real-Time Text to Speech")

text = st.text_area("Enter Text to Speak", "Hey how are you?")
description = st.text_input("Speaker Description", "A male speaker with a deep voice and animated speech")

if st.button("Generate Speech"):
    if not text or not description:
        st.error("Please provide both text and description.")
    else:
        with st.spinner("Generating speech..."):
            request = massanger_pb2.TextRequest(text=text, description=description)
            response = stub.GenerateSpeech(request)
            audio_file = "output.wav"
            with open(audio_file, "wb") as f:
                f.write(response.audio)
            st.success("Speech generated!")
            st.audio(audio_file, format="audio/wav")
