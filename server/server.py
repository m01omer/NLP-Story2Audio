import grpc
from grpc import aio
import asyncio
import logging
import torch
import io
import soundfile as sf
import os
import sys
import socket

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_loader import load_model
from generated import tts_service_pb2, tts_service_pb2_grpc

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("server_debug.log"),
        logging.StreamHandler()
    ]
)

# Docker detection function
def is_running_in_docker():
    """Check if the application is running inside a Docker container."""
    # Method 1: Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    
    # Method 2: Check cgroup
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return any('docker' in line for line in f)
    except:
        pass
    
    # Method 3: Check environment variable (can be set in Docker compose or Dockerfile)
    return os.environ.get('RUNNING_IN_DOCKER', '').lower() in ('true', '1', 't')

# Check if port is available
def is_port_available(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', port))
        sock.close()
        return True
    except:
        return False

# Load model and resources
print("Loading TTS model...")
model, prompt_tokenizer, description_tokenizer = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()

# Simple lock for model access
model_lock = asyncio.Lock()

class TTSServicer(tts_service_pb2_grpc.TTSServicer):

    async def GenerateSpeech(self, request, context):
        try:
            logging.info(f"Received request with text: '{request.text[:20]}...'")
            
            if not request.text.strip() or not request.description.strip():
                logging.warning("Received empty text or description")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text and description must not be empty.")
                return tts_service_pb2.AudioResponse()

            # Tokenization
            desc_input = description_tokenizer(request.description, return_tensors="pt").to(device)
            prompt_input = prompt_tokenizer(request.text, return_tensors="pt").to(device)

            # Use lock for model access
            async with model_lock:
                logging.info("Generating audio...")
                with torch.no_grad():
                    generated_audio = model.generate(
                        input_ids=desc_input.input_ids,
                        attention_mask=desc_input.attention_mask,
                        prompt_input_ids=prompt_input.input_ids,
                        prompt_attention_mask=prompt_input.attention_mask
                    )

            audio = generated_audio.cpu().numpy().squeeze()
            if audio.size == 0:
                raise ValueError("Empty audio output.")

            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, model.config.sampling_rate, format="WAV")
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            logging.info(f"Successfully generated audio of size: {len(audio_bytes)} bytes")
            return tts_service_pb2.AudioResponse(audio=audio_bytes)

        except Exception as e:
            log_msg = f"Error in GenerateSpeech: {e}"
            logging.error(log_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tts_service_pb2.AudioResponse()

async def serve():
    port = 50051
    
    # Check if running in Docker
    in_docker = is_running_in_docker()
    
    # Adjust port check based on environment
    if not in_docker and not is_port_available(port):
        logging.error(f"Port {port} is already in use! Please close the application using it or choose a different port.")
        print(f"❌ ERROR: Port {port} is already in use!")
        return
    
    # Server options
    server_options = [
        ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
        ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
    ]
    
    # Create server
    server = aio.server(options=server_options)
    tts_service_pb2_grpc.add_TTSServicer_to_server(TTSServicer(), server)
    
    # Get hostname
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    # Bind server - always listen on all interfaces
    server_address = "0.0.0.0:50051"
    server.add_insecure_port(server_address)
    
    # Start server
    await server.start()
    
    # Log server information with Docker-specific guidance
    if in_docker:
        print(f"🐳 gRPC TTS Server running in Docker container:")
        print(f"   - Container hostname: {hostname}")
        print(f"   - Container IP: {ip_address}")
        print(f"   - Port: 50051 (exposed according to your Docker configuration)")
        print("When connecting from the host, use the Docker host IP and the exposed port.")
        print("When connecting from another container, use the container name or network alias.")
    else:
        print(f"🚀 gRPC TTS Server running locally on:")
        print(f"   - localhost:50051")
        print(f"   - {ip_address}:50051")
        print(f"   - {hostname}:50051")
        print(f"Machine name: {hostname}")
        print("Try any of these addresses in your UI config.")
    
    # Wait for shutdown
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down server...")
        await server.stop(5)

if __name__ == "__main__":
    asyncio.run(serve())