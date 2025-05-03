# Modified Streamlit app with vibrant UI, animations, intro screen, and loading effect

import streamlit as st
import grpc
import os
import sys
import uuid
import time
import socket
from streamlit_lottie import st_lottie
import json

# Load Lottie animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Add parent directory to path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generated import tts_service_pb2, tts_service_pb2_grpc

# Docker detection function
def is_running_in_docker():
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return any('docker' in line for line in f)
    except:
        pass
    return os.environ.get('RUNNING_IN_DOCKER', '').lower() in ('true', '1', 't')

# App config
st.set_page_config(page_title="TTS Demo", layout="wide", initial_sidebar_state="collapsed")

# Intro screen
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
    lottie_intro = load_lottiefile(r"frontend\animations\welcome.json")
    st_lottie(lottie_intro, height=1000)
    st.markdown("""
        <script>
            window.scrollTo({
                top: 99999,
                left: 0,
                behavior: 'smooth'
            });
        </script>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Welcome to the Text-to-Speech Demo! 🎙️</h2>", unsafe_allow_html=True)
    with st.spinner("Loading app..."):
        time.sleep(5)
    st.session_state.loaded = True
    st.rerun()


# # Main content
# st.title("Text to Speech")

# Collapsible sidebar
with st.sidebar:
    with st.expander("🔧 Server Settings", expanded=False):
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except:
            hostname = "Unknown"
            local_ip = "127.0.0.1"

        in_docker = is_running_in_docker()
        server_host_env = os.environ.get('TTS_SERVER_HOST', '')
        server_port_env = os.environ.get('TTS_SERVER_PORT', '50051')

        address_options = {}
        if in_docker:
            st.info("🐳 Running in Docker container")
            if server_host_env:
                address_options["Environment setting"] = f"{server_host_env}:{server_port_env}"
            address_options.update({
                "Service name (Docker)": f"tts-server:{server_port_env}",
                "Container IP": f"{local_ip}:{server_port_env}",
            })
        else:
            address_options.update({
                "localhost": "localhost:50051",
                "IP address": f"{local_ip}:50051",
                "Computer name": f"{hostname}:50051",
            })

        address_options["Custom"] = "custom"
        selected_option = st.radio("Server address:", options=list(address_options.keys()))

        if selected_option == "Custom":
            placeholder = "tts-server:50051" if in_docker else "localhost:50051"
            custom_address = st.text_input("Enter server address:", placeholder)
            server_address = custom_address
        else:
            server_address = address_options[selected_option]

        clear_cache = st.checkbox("Force new connection", value=True)

        if in_docker:
            st.caption("⚙️ Docker network settings")
            st.code(f"""
HOSTNAME: {hostname}
LOCAL IP: {local_ip}
ENV VARS: {'Set' if server_host_env else 'Not set'}
            """)

        if st.button("Test Connection"):
            with st.status("Testing connection..."):
                try:
                    channel = grpc.insecure_channel(server_address)
                    try:
                        grpc.channel_ready_future(channel).result(timeout=3)
                        st.success(f"✅ Connected to {server_address}!")
                        channel.close()
                    except grpc.FutureTimeoutError:
                        st.error(f"❌ Connection timeout to {server_address}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

# Speaker mapping
speaker_map = {
    "Default" : "Default",
    "bdl (US male)"        : "cmu_us_bdl_arctic-wav-arctic_a0009",
    "clb (US female)"      : "cmu_us_clb_arctic-wav-arctic_a0144",
    "ksp (Indian male)"    : "cmu_us_ksp_arctic-wav-arctic_b0087",
    "rms (US male)"        : "cmu_us_rms_arctic-wav-arctic_b0353",
    "slt (US female)"      : "cmu_us_slt_arctic-wav-arctic_a0508",
}

col1, col2 = st.columns(2)

with col1:
    lottie_voice = load_lottiefile(r"frontend\animations\voice.json")
    st_lottie(lottie_voice, speed=1, height=250, key="voice")
    text = st.text_area("Enter text to speak:", "Hello, this is a test of the text to speech system.")
    description_label = st.selectbox("Choose a voice:", options=list(speaker_map.keys()))
    description = speaker_map[description_label]

if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

with col2:
    lottie_speak = load_lottiefile("frontend/animations/speak.json")

    if in_docker:
        st.info("Make sure the server container is running before generating speech.")
    else:
        st.info("Make sure the server is running before generating speech.")

    if st.button("🔊 Generate Speech", use_container_width=True):
        if not text.strip() or not description.strip():
            st.error("Please enter both text and description.")
        else:
            st.session_state.is_generating = True
            placeholder = st.empty()

            # Show animation + text (centered horizontally)
            with placeholder.container():
                col1, col2 = st.columns([1, 20])
                with col1:
                    st_lottie(lottie_speak, key="lottie_spinner")

                with col2:
                    st.markdown("")

            try:
                channel = grpc.insecure_channel(server_address)
                timeout = 10 if in_docker else 5
                grpc.channel_ready_future(channel).result(timeout=timeout)
                stub = tts_service_pb2_grpc.TTSStub(channel)
                request = tts_service_pb2.TextRequest(text=text, description=description)
                start_time = time.time()
                response = stub.GenerateSpeech(request, timeout=60)
                generation_time = time.time() - start_time

                if response and response.audio:
                    audio_file = f"audio_{uuid.uuid4().hex[:8]}.wav"
                    with open(audio_file, "wb") as f:
                        f.write(response.audio)

                    placeholder.empty()  # remove the loader
                    st.success(f"✅ Generated in {generation_time:.1f} seconds!")
                    st.audio(audio_file, format="audio/wav")

                    with open(audio_file, "rb") as f:
                        st.download_button("⬇️ Download audio", f, file_name="tts_output.wav", mime="audio/wav")
                else:
                    placeholder.empty()
                    st.error("Received empty response from server.")
            except grpc.FutureTimeoutError:
                placeholder.empty()
                st.error(f"Connection timeout to {server_address}")
            except Exception as e:
                placeholder.empty()
                st.error(f"Error: {str(e)}")
            finally:
                channel.close()
                st.session_state.is_generating = False

st.divider()

# Footer info
if in_docker:
    st.caption("TTS Demo Application (Docker Environment)")
else:
    st.caption("TTS Demo Application (Local Environment)")
