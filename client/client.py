# client/client.py
import grpc
from generated import massanger_pb2, massanger_pb2_grpc

def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = massanger_pb2_grpc.TTSStub(channel)

    request = massanger_pb2.TextRequest(
        text="Hey how are you?",
        description=(
            "A male speaker with a deep voice "
            "and animated speech with a moderate speed and pitch."
        )
    )

    response = stub.GenerateSpeech(request)

    with open("tts_output.wav", "wb") as f:
        f.write(response.audio)

    print("Audio saved to tts_output.wav")

if __name__ == "__main__":
    run()
