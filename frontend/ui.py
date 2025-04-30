# ui.py

import streamlit as st
import grpc
from generated import tts_service_pb2, tts_service_pb2_grpc
import os
# Allow switching between local and Docker gRPC addresses
DEFAULT_GRPC_ADDRESS = "localhost:50051"
GRPC_ADDRESS = st.secrets.get("grpc_address", os.getenv("GRPC_ADDRESS", DEFAULT_GRPC_ADDRESS))

# Setup gRPC connection
channel = grpc.insecure_channel(GRPC_ADDRESS)
stub = tts_service_pb2_grpc.TTSStub(channel)

st.title("🗣️ Real-Time Text to Speech")

text = st.text_area("Enter Text to Speak", "Hey how are you?")
description = st.text_input("Speaker Description", "A male speaker with a deep voice and animated speech")

if st.button("Generate Speech"):
	if not text.strip() or not description.strip():
		st.error("Text and description cannot be empty.")
	else:
		with st.spinner("Generating speech..."):
			request = tts_service_pb2.TextRequest(text=text, description=description)
			try:
				response = stub.GenerateSpeech(request)

				if not response.audio:
					st.error("No audio was returned. Check your input or try again.")
				else:
					audio_file = "output.wav"
					with open(audio_file, "wb") as f:
						f.write(response.audio)
					st.success("Speech generated successfully!")
					st.audio(audio_file, format="audio/wav")

			except grpc.RpcError as e:
				st.error(f"gRPC error: {e.code().name} - {e.details()}")
			except Exception as ex:
				st.error(f"Unexpected error: {ex}")
