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
import re
import numpy as np  
from datasets import load_dataset
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_loader import load_model
from generated import tts_service_pb2, tts_service_pb2_grpc
from process import TextPreprocessor

# Configuration
DEVICE_MODE = 'cuda'  # Change this to 'auto' to use GPU if available
OUTPUT_DIR = "generated_audio"  # Directory to save generated audio files

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

if DEVICE_MODE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide all CUDA devices
    if 'torch' in sys.modules:
        import torch
        torch.cuda.is_available = lambda: False 


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Using CPU")

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
    
    # Method 3: Check environment variable
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

# Load Speaker Embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embedding = torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)
def get_speaker_embedding(speaker_name: str):
    for e in embeddings_dataset:
        if e["filename"].split("/")[0].lower() == speaker_name.lower():
            return torch.tensor(e["xvector"]).unsqueeze(0)
    logging.warning(f"Speaker '{speaker_name}' not found, using default.")
    return torch.tensor(embeddings_dataset[0]["xvector"]).unsqueeze(0)


# Initialize components
text_preprocessor = TextPreprocessor()
model, tokenizer, processor, vocoder, device = load_model()

# Simple lock for model access
model_lock = asyncio.Lock()

def save_audio_to_file(audio_data, sample_rate, description):
    """Save audio bytes to a file with timestamp and description."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean description for filename (remove special chars)
    clean_desc = re.sub(r'[^\w\s-]', '', description).strip().replace(' ', '_')
    if not clean_desc:
        clean_desc = "tts_output"
        
    filename = f"{clean_desc}_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        sf.write(filepath, audio_data, sample_rate, format="WAV")
        logging.info(f"Audio saved to {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"Failed to save audio file: {e}")
        return None

class TTSServicer(tts_service_pb2_grpc.TTSServicer):
    async def GenerateSpeech(self, request, context):
        start_time = time.time()
        try:
            original_text = request.text
            original_desc = request.description
            speaker_embedding = get_speaker_embedding(original_desc)
            logging.info(f"Received request with text: '{original_text[:20]}...'")
            
            # Validate input
            if not original_text.strip() or not original_desc.strip():
                logging.warning("Received empty text or description")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text and description must not be empty.")
                return tts_service_pb2.AudioResponse()

            # Add timeout handling for long requests
            if len(original_text) > 5000:
                logging.warning(f"Text too long: {len(original_text)} chars")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text exceeds maximum allowed length (5000 characters).")
                return tts_service_pb2.AudioResponse()

            # Preprocess the text and description
            preprocess_start = time.time()
            processed_text = text_preprocessor.preprocess_text(original_text)
            processed_desc = text_preprocessor.preprocess_description(original_desc)
            preprocess_time = time.time() - preprocess_start
            logging.info(f"Preprocessing completed in {preprocess_time:.3f}s")
            
            # Split into chunks BEFORE tokenization to avoid token length errors
            max_chars_per_chunk = 300  # Conservative estimate, adjust based on your model's token-to-char ratio
            chunks = []
            
            # Split text into sentences
            sentences = re.split(r'([.!?])', processed_text)
            current_chunk = ""
            
            for i in range(0, len(sentences), 2):
                if i+1 < len(sentences):
                    sentence = sentences[i] + sentences[i+1]
                else:
                    sentence = sentences[i]
                    
                # If adding this sentence would make the chunk too long, save current chunk and start a new one
                if len(current_chunk) + len(sentence) > max_chars_per_chunk and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += sentence
            
            # Add the last chunk if it's not empty
            if current_chunk:
                chunks.append(current_chunk)
            
            logging.info(f"Split text into {len(chunks)} chunks for processing")
            
            # Use lock for model access
            async with model_lock:
                # Process chunks and generate audio
                audio_chunks = []
                
                for chunk_idx, chunk in enumerate(chunks):
                    logging.info(f"Processing chunk {chunk_idx+1}/{len(chunks)} with {len(chunk)} chars")
                    
                    # Tokenize each chunk separately
                    token_start = time.time()
                    chunk_inputs = processor(text=chunk, return_tensors="pt").to(device)
                    token_time = time.time() - token_start
                    
                    # Check if chunk is still too long after tokenization
                    token_length = chunk_inputs["input_ids"].shape[1]
                    logging.info(f"Chunk {chunk_idx+1} tokenized to {token_length} tokens in {token_time:.3f}s")
                    
                    if token_length > 600:
                        logging.warning(f"Chunk {chunk_idx+1} still too long: {token_length} tokens > 600. Trimming...")
                        chunk_inputs["input_ids"] = chunk_inputs["input_ids"][:, :600]
                    
                    # Generate speech for this chunk
                    infer_start = time.time()
                    with torch.no_grad():
                        chunk_audio = model.generate_speech(
                            chunk_inputs["input_ids"],
                            speaker_embeddings=speaker_embedding.to(device),
                            vocoder=vocoder
                        ).cpu().numpy()
                    
                    infer_time = time.time() - infer_start
                    logging.info(f"Chunk {chunk_idx+1} audio generated in {infer_time:.3f}s")
                    
                    # Add to audio chunks
                    audio_chunks.append(chunk_audio)
                    
                    # Clear memory after each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Combine all audio chunks
                if audio_chunks:
                    audio = np.concatenate(audio_chunks)
                    logging.info(f"Combined {len(audio_chunks)} audio chunks into final audio of length {len(audio)}")
                else:
                    logging.error("No audio chunks were generated")
                    raise ValueError("Failed to generate any audio chunks")

            if audio.size == 0:
                raise ValueError("Empty audio output.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            audio_file_path = save_audio_to_file(audio, processor.feature_extractor.sampling_rate, original_desc)
            print(f"🔊 Audio saved to: {audio_file_path}")
            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, processor.feature_extractor.sampling_rate, format="WAV")
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            # Save to file - NEW CODE
            audio_file_path = save_audio_to_file(audio, processor.feature_extractor.sampling_rate, original_desc)
            
            total_time = time.time() - start_time
            logging.info(f"Successfully generated audio of size: {len(audio_bytes)} bytes in {total_time:.3f}s")
            
            # Add file path to the response - NEW CODE
            response = tts_service_pb2.AudioResponse(audio=audio_bytes)
            if audio_file_path:
                context.set_trailing_metadata([
                    ('file_path', audio_file_path)
                ])
                print(f"🔊 Audio saved to: {audio_file_path}")
            
            return response

        except Exception as e:
            total_time = time.time() - start_time
            log_msg = f"Error in GenerateSpeech after {total_time:.3f}s: {e}"
            logging.error(log_msg)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tts_service_pb2.AudioResponse()

async def serve():
    # Set PyTorch memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    if DEVICE_MODE == 'auto' and torch.cuda.is_available():
        try:
            torch.cuda.memory._set_allocator_settings('garbage_collection_threshold:0.8')
            logging.info("Enabled aggressive CUDA memory garbage collection")
        except:
            logging.info("Could not set advanced memory allocator settings")
    
    # Set smaller default tensors for inference
    torch.set_default_tensor_type(torch.FloatTensor)
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
    
    # Log server information
    print(f"🔊 SpeechT5 TTS model loaded successfully")
    print(f"📁 Audio files will be saved to: {os.path.abspath(OUTPUT_DIR)}")
    
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