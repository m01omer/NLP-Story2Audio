# server/server.py

import grpc
from grpc import aio
import io
import soundfile as sf
import torch
import asyncio
import logging
from model_loader import load_model
import sys
import os

# Automatically add the parent directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'generated')))
from generated import tts_service_pb2
from generated import tts_service_pb2_grpc

# Setup logging
logging.basicConfig(filename="server_errors.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and tokenizers
model, prompt_tokenizer, description_tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Async lock for model inference
model_lock = asyncio.Lock()

class TTSServicer(tts_service_pb2_grpc.TTSServicer):

    async def GenerateSpeech(self, request, context):
        try:
            # Input validation
            if not request.text.strip() or not request.description.strip():
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text or description cannot be empty.")
                return tts_service_pb2.AudioResponse()

            # Tokenize inputs
            desc_inputs = description_tokenizer(request.description, return_tensors="pt").to(device)
            prompt_inputs = prompt_tokenizer(request.text, return_tensors="pt").to(device)

            # Model inference (async + thread-safe)
            async with model_lock:
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=desc_inputs.input_ids,
                        attention_mask=desc_inputs.attention_mask,
                        prompt_input_ids=prompt_inputs.input_ids,
                        prompt_attention_mask=prompt_inputs.attention_mask
                    )

            # Convert to numpy
            audio = gen.cpu().numpy().squeeze()
            if audio.size == 0:
                raise ValueError("Generated audio is empty.")

            # Convert to WAV
            buf = io.BytesIO()
            sf.write(buf, audio, model.config.sampling_rate, format="WAV")
            buf.seek(0)

            return tts_service_pb2.AudioResponse(audio=buf.read())

        except Exception as e:
            log_msg = f"Error in GenerateSpeech: {e}"
            logging.error(log_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tts_service_pb2.AudioResponse()

async def serve():
    server = aio.server()
    tts_service_pb2_grpc.add_TTSServicer_to_server(TTSServicer(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    print("Async TTS Server started on port 50051...")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
