# server/server.py
import grpc
from concurrent import futures
import io
import soundfile as sf
import torch

from model_loader import load_model

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'generated')))

from generated import massanger_pb2
from generated import massanger_pb2_grpc

model, prompt_tokenizer, description_tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"

class TTSServicer(massanger_pb2_grpc.TTSServicer):
    def GenerateSpeech(self, request, context):
        try:
            # Tokenize description and text inputs
            desc_inputs = description_tokenizer(request.description, return_tensors="pt").to(device)
            prompt_inputs = prompt_tokenizer(request.text, return_tensors="pt").to(device)

            # Generate speech
            with torch.no_grad():
                gen = model.generate(
                    input_ids=desc_inputs.input_ids,
                    attention_mask=desc_inputs.attention_mask,
                    prompt_input_ids=prompt_inputs.input_ids,
                    prompt_attention_mask=prompt_inputs.attention_mask
                )

            # Convert generated speech to WAV
            audio = gen.cpu().numpy().squeeze()
            buf = io.BytesIO()
            sf.write(buf, audio, model.config.sampling_rate, format="WAV")
            buf.seek(0)

            # Return the audio response
            return massanger_pb2.AudioResponse(audio=buf.read())

        except Exception as e:
            # Handle errors
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return massanger_pb2.AudioResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    massanger_pb2_grpc.add_TTSServicer_to_server(TTSServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
