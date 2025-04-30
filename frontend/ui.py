import streamlit as st
import grpc
import os
import sys
import uuid
import time
import socket

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generated import tts_service_pb2, tts_service_pb2_grpc

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

st.set_page_config(page_title="TTS Demo", layout="wide")
st.title("🗣️ Text to Speech Demo")

# Server connection options
st.sidebar.header("Server Settings")

# Try to get local hostname and IP
try:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
except:
    hostname = "Unknown"
    local_ip = "127.0.0.1"

# Check if running in Docker
in_docker = is_running_in_docker()

# Get Docker-specific environment variables if available
server_host_env = os.environ.get('TTS_SERVER_HOST', '')
server_port_env = os.environ.get('TTS_SERVER_PORT', '50051')

# Server address options (with Docker-aware defaults)
address_options = {}

if in_docker:
    # In Docker, prioritize container communication
    st.sidebar.info("🐳 Running in Docker container")
    
    # Use environment variables if provided
    if server_host_env:
        address_options["Environment setting"] = f"{server_host_env}:{server_port_env}"
    
    # Add container network options
    address_options.update({
        "Service name (Docker)": f"tts-server:{server_port_env}",  # Assuming service is named 'tts-server'
        "Container IP": f"{local_ip}:{server_port_env}",
    })
else:
    # Local environment options
    address_options.update({
        "localhost": "localhost:50051",
        "IP address": f"{local_ip}:50051",
        "Computer name": f"{hostname}:50051",
    })

# Always include custom option
address_options["Custom"] = "custom"

selected_option = st.sidebar.radio("Server address:", options=list(address_options.keys()))

# Custom address input
if selected_option == "Custom":
    # Provide Docker-specific hint
    placeholder = "tts-server:50051" if in_docker else "localhost:50051"
    custom_address = st.sidebar.text_input("Enter server address:", placeholder)
    server_address = custom_address
else:
    server_address = address_options[selected_option]

# Force clear channel cache
clear_cache = st.sidebar.checkbox("Force new connection", value=True)

# Environment info
if in_docker:
    st.sidebar.divider()
    st.sidebar.caption("⚙️ Docker network settings")
    st.sidebar.code(f"""
HOSTNAME: {hostname}
LOCAL IP: {local_ip}
ENV VARS: {'Set' if server_host_env else 'Not set'}
    """)

# Test connection button
if st.sidebar.button("Test Connection"):
    with st.sidebar.status("Testing connection..."):
        try:
            # Create channel
            channel = grpc.insecure_channel(server_address)
            
            # Try to connect with timeout
            try:
                grpc.channel_ready_future(channel).result(timeout=3)
                st.sidebar.success(f"✅ Connected to {server_address}!")
                channel.close()
            except grpc.FutureTimeoutError:
                st.sidebar.error(f"❌ Connection timeout to {server_address}")
        except Exception as e:
            st.sidebar.error(f"❌ Connection error: {str(e)}")

# Main UI
col1, col2 = st.columns(2)

with col1:
    text = st.text_area("Enter text to speak:", "Hello, this is a test of the text to speech system.")
    description = st.text_input("Voice description:", "A friendly female voice speaking clearly and calmly.")

with col2:
    # Adjust message based on environment
    if in_docker:
        st.info("Make sure the server container is running before generating speech.")
    else:
        st.info("Make sure the server is running before generating speech.")
    
    if st.button("Generate Speech", use_container_width=True):
        if not text.strip() or not description.strip():
            st.error("Please enter both text and description.")
        else:
            with st.spinner("Generating speech..."):
                try:
                    # Create a fresh channel each time if requested
                    channel = grpc.insecure_channel(server_address)
                    
                    # Set timeout - slightly longer timeout for Docker environments
                    timeout = 10 if in_docker else 5
                    try:
                        grpc.channel_ready_future(channel).result(timeout=timeout)
                        
                        # Create request
                        stub = tts_service_pb2_grpc.TTSStub(channel)
                        request = tts_service_pb2.TextRequest(
                            text=text, 
                            description=description
                        )
                        
                        # Send request
                        start_time = time.time()
                        response = stub.GenerateSpeech(request, timeout=60)
                        generation_time = time.time() - start_time
                        
                        # Process response
                        if response and response.audio:
                            # Save to file
                            audio_file = f"audio_{uuid.uuid4().hex[:8]}.wav"
                            with open(audio_file, "wb") as f:
                                f.write(response.audio)
                            
                            # Display success
                            st.success(f"✅ Generated in {generation_time:.1f} seconds!")
                            
                            # Play audio
                            st.audio(audio_file, format="audio/wav")
                            
                            # Download button
                            with open(audio_file, "rb") as f:
                                st.download_button(
                                    "Download audio",
                                    f,
                                    file_name="tts_output.wav",
                                    mime="audio/wav"
                                )
                        else:
                            st.error("Received empty response from server.")
                            
                    except grpc.FutureTimeoutError:
                        st.error(f"Connection timeout to {server_address}")
                    finally:
                        channel.close()
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.divider()

# Environment indicator
if in_docker:
    st.caption("TTS Demo Application (Docker Environment)")
else:
    st.caption("TTS Demo Application (Local Environment)")