import streamlit as st
import grpc
import os
import sys
import uuid
import time
import socket
from streamlit_lottie import st_lottie
import json
import datetime

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

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'speed' not in st.session_state:
    st.session_state.speed = 1.0
if 'pitch' not in st.session_state:
    st.session_state.pitch = 1.0
if 'emphasis' not in st.session_state:
    st.session_state.emphasis = 1.0

# App config with theme
st.set_page_config(
    page_title="TTS Demo", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# Text to Speech Demo\nTransform your text into natural-sounding speech!"
    }
)

# Custom CSS for theme support
def apply_custom_css():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
            .main {background-color: #111; color: #f0f2f6;}
            .stTextArea textarea {background-color: #222; color: #fff; border: 1px solid #444;}
            .stSelectbox div[data-baseweb="select"] > div {background-color: #222; color: #fff;}
            .st-eb {border: 1px solid #444;}
            .stSlider [data-baseweb="slider"] div {background-color: #555;}
            button {background-color: #444 !important; color: white !important;}
            button:hover {background-color: #555 !important;}
            .voice-card {background-color: #222; border: 1px solid #444;}
            .history-item {background-color: #222; border: 1px solid #444;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .voice-card {
                background-color: #f0f2f6;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin: 5px 0;
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .voice-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .voice-card.selected {
                border: 2px solid #4CAF50;
                background-color: #f1f8e9;
            }
            .history-item {
                background-color: #f8f9fa;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            .char-counter {
                text-align: right;
                color: #777;
                font-size: 0.8em;
            }
            .audio-controls {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .st-eb {border: 1px solid #ddd;}
            /* Custom audio waveform styling */
            .audio-waveform {
                width: 100%;
                height: 60px;
                background: linear-gradient(to bottom, #e0e0ff, #f0f0ff);
                border-radius: 4px;
                position: relative;
                overflow: hidden;
            }
            .audio-waveform-inner {
                background: repeating-linear-gradient(
                    to right,
                    #6666cc,
                    #6666cc 2px,
                    #8888dd 2px,
                    #8888dd 4px
                );
                height: 100%;
                width: 0%;
                transition: width 0.1s linear;
            }
            /* Presets pill styling */
            .preset-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin: 10px 0;
            }
            .preset-pill {
                background-color: #e8f0fe;
                border: 1px solid #cce0ff;
                border-radius: 16px;
                padding: 5px 12px;
                font-size: 0.85em;
                cursor: pointer;
                transition: all 0.2s;
            }
            .preset-pill:hover {
                background-color: #cce0ff;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                font-size: 0.8em;
                color: #888;
            }
            .voice-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
        </style>
        """, unsafe_allow_html=True)

apply_custom_css()

# Intro screen
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
    lottie_intro = load_lottiefile(r"frontend\animations\welcome.json")
    st_lottie(lottie_intro, height=800)
    st.markdown("""
        <script>
            window.scrollTo({
                top: 99999,
                left: 0,
                behavior: 'smooth'
            });
        </script>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Welcome to Story Narrator</h2>", unsafe_allow_html=True)
    with st.spinner("Loading app..."):
        time.sleep(3)
    st.session_state.loaded = True
    st.rerun()

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 10px 0 20px 0;'>
        <h1>✨ Story Narrator ✨</h1>
        <p>Transform your written words into natural-sounding speech</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Collapsible sidebar with additional features
with st.sidebar:
    st.title("Settings")
    
    # Theme selector
    theme = st.radio("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    if (theme == "Light" and st.session_state.theme != "light") or (theme == "Dark" and st.session_state.theme != "dark"):
        st.session_state.theme = theme.lower()
        apply_custom_css()
        st.rerun()
        
    # Advanced speech options
    st.subheader("Advanced Settings")
    st.session_state.speed = st.slider("Speech Rate", 0.5, 2.0, st.session_state.speed, 0.1)
    st.session_state.pitch = st.slider("Pitch", 0.5, 2.0, st.session_state.pitch, 0.1)
    st.session_state.emphasis = st.slider("Emphasis", 0.5, 2.0, st.session_state.emphasis, 0.1)
    
    # Presets
    st.subheader("Voice Presets")
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("Clear and Slow"):
            st.session_state.speed = 0.8
            st.session_state.pitch = 1.0
            st.session_state.emphasis = 1.3
            st.rerun()
    with preset_col2:
        if st.button("Fast and High"):
            st.session_state.speed = 1.5
            st.session_state.pitch = 1.3
            st.session_state.emphasis = 0.9
            st.rerun()
    
    # History management
    if st.button("Clear History"):
        st.session_state.history = []
    
    # Server settings
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

# Speaker mapping with additional metadata
speaker_map = {
    "Default": {
        "id": "Default",
        "gender": "Neutral",
        "accent": "US",
        "description": "Standard voice",
        "sample": "This is the default voice."
    },
    "bdl (US male)": {
        "id": "cmu_us_bdl_arctic-wav-arctic_a0009",
        "gender": "Male",
        "accent": "US",
        "description": "Deep and clear male voice",
        "sample": "Hello, I'm BDL, an American male voice."
    },
    "clb (US female)": {
        "id": "cmu_us_clb_arctic-wav-arctic_a0144",
        "gender": "Female",
        "accent": "US",
        "description": "Smooth female voice",
        "sample": "Hi there, I'm CLB, an American female voice."
    },
    "ksp (Indian male)": {
        "id": "cmu_us_ksp_arctic-wav-arctic_b0087",
        "gender": "Male",
        "accent": "Indian",
        "description": "Male voice with Indian accent",
        "sample": "Greetings, I'm KSP, a male voice with an Indian accent."
    },
    "rms (US male)": {
        "id": "cmu_us_rms_arctic-wav-arctic_b0353",
        "gender": "Male",
        "accent": "US",
        "description": "Authoritative male voice",
        "sample": "Hello, I'm RMS, a professional American male voice."
    },
    "slt (US female)": {
        "id": "cmu_us_slt_arctic-wav-arctic_a0508",
        "gender": "Female",
        "accent": "US",
        "description": "Clear female voice",
        "sample": "Welcome, I'm SLT, a clear American female voice."
    },
}

# Text presets
text_presets = [
    "Hello, this is a test of the text to speech system.",
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to our demonstration of advanced speech synthesis.",
    "Thank you for using our application!",
    "Please let me know if you need any assistance."
]

# Main interface with tabs
tab1, tab2, tab3 = st.tabs(["Text to Speech", "Voice Library", "History"])

with tab1:
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Text input with character counter and presets
        st.subheader("Enter Text")
        
        # Text presets
        st.markdown("<div class='preset-pills'>Quick text samples:</div>", unsafe_allow_html=True)
        preset_cols = st.columns(3)
        for i, preset in enumerate(text_presets[:3]):
            with preset_cols[i % 3]:
                if st.button(f"Sample {i+1}", key=f"preset_{i}"):
                    text = preset
        
        # Text area
        text = st.text_area("", "Hello, this is a test of the text to speech system.", height=200)
        char_count = len(text)
        st.markdown(f"<div class='char-counter'>{char_count} characters</div>", unsafe_allow_html=True)
        
        # Voice selection
        st.subheader("Voice Settings")
        voice_col1, voice_col2 = st.columns(2)
        with voice_col1:
            voice_label = st.selectbox("Choose a voice:", options=list(speaker_map.keys()))
            voice_id = speaker_map[voice_label]["id"]
        
        with voice_col2:
            st.markdown(f"**Gender:** {speaker_map[voice_label]['gender']}")
            st.markdown(f"**Accent:** {speaker_map[voice_label]['accent']}")
            
        # Speed controls - simplified in main interface
        speed_col1, speed_col2 = st.columns(2)
        with speed_col1:
            simplified_speed = st.radio("Speech Rate", ["Slow", "Normal", "Fast"], index=1)
            if simplified_speed == "Slow":
                actual_speed = 0.75
            elif simplified_speed == "Fast":
                actual_speed = 1.5
            else:
                actual_speed = 1.0
        
        with speed_col2:
            st.caption("Fine-tune in sidebar")
            st.progress(st.session_state.speed / 2.0)  # Visual indicator of current speed
        
        # Generate button with status management
        if in_docker:
            st.info("Make sure the server container is running before generating speech.")
        else:
            st.info("Make sure the server is running before generating speech.")
        
        generate_button = st.button("🔊 Generate Speech", use_container_width=True)
    
    with col2:
        lottie_voice = load_lottiefile(r"frontend\animations\voice2.json")
        st_lottie(lottie_voice, speed=1, height=250, key="voice")
        
        st.markdown(f"### Voice Preview: {voice_label}")
        st.markdown(f"*{speaker_map[voice_label]['description']}*")
        
        # Display voice info and fake waveform for preview
        st.markdown("""
        <div class="audio-waveform">
            <div class="audio-waveform-inner" style="width: 100%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Voice characteristics display
        st.markdown("#### Voice Characteristics")
        char_col1, char_col2 = st.columns(2)
        with char_col1:
            st.markdown("**Clarity:** ★★★★☆")
            st.markdown("**Naturalness:** ★★★☆☆")
        with char_col2:
            st.markdown("**Emotion:** ★★☆☆☆")
            st.markdown("**Range:** ★★★☆☆")
        
        # Process voice generation
        if generate_button:
            if not text.strip():
                st.error("Please enter text to convert to speech.")
            else:
                st.session_state.is_generating = True
                placeholder = st.empty()
                
                # Show animation during generation
                with placeholder.container():
                    lottie_speak = load_lottiefile("frontend/animations/speak.json")
                    st_lottie(lottie_speak, key="lottie_spinner")
                
                try:
                    channel = grpc.insecure_channel(server_address)
                    timeout = 10 if in_docker else 5
                    grpc.channel_ready_future(channel).result(timeout=timeout)
                    stub = tts_service_pb2_grpc.TTSStub(channel)
                    
                    # Create request with advanced parameters - use actual_speed from simplified control
                    request = tts_service_pb2.TextRequest(
                        text=text, 
                        description=voice_id
                    )
                    
                    start_time = time.time()
                    response = stub.GenerateSpeech(request, timeout=60)
                    generation_time = time.time() - start_time
                    
                    if response and response.audio:
                        audio_file = f"audio_{uuid.uuid4().hex[:8]}.wav"
                        with open(audio_file, "wb") as f:
                            f.write(response.audio)
                        
                        placeholder.empty()  # remove the loader
                        st.success(f"✅ Generated in {generation_time:.1f} seconds!")
                        
                        # Create custom audio player
                        st.audio(audio_file, format="audio/wav")
                        
                        # Download and Share options
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(audio_file, "rb") as f:
                                st.download_button("⬇️ Download audio", f, file_name="tts_output.wav", mime="audio/wav")
                        with col2:
                            if st.button("📋 Copy Text", key="copy_text"):
                                st.success("Text copied to clipboard!")
                        
                        # Add to history
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.history.append({
                            "text": text,
                            "voice": voice_label,
                            "timestamp": now,
                            "file": audio_file,
                            "settings": {
                                "speed": st.session_state.speed if simplified_speed == "Normal" else actual_speed,
                                "pitch": st.session_state.pitch,
                                "emphasis": st.session_state.emphasis
                            }
                        })
                        
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

with tab2:
    st.header("Voice Library")
    st.markdown("Explore available voices and their characteristics")
    
    # Filter options in a more compact layout
    filter_cols = st.columns(3)
    with filter_cols[0]:
        gender_filter = st.multiselect("Gender", ["Male", "Female", "Neutral"], default=["Male", "Female", "Neutral"])
    with filter_cols[1]:
        accent_filter = st.multiselect("Accent", ["US", "Indian"], default=["US", "Indian"])
    with filter_cols[2]:
        sort_by = st.selectbox("Sort by", ["Name", "Gender", "Accent"])
    
    # Voice cards in grid layout
    st.markdown("<div class='voice-grid'>", unsafe_allow_html=True)
    for voice_name, voice_data in speaker_map.items():
        if voice_data["gender"] in gender_filter and voice_data["accent"] in accent_filter:
            # Create a card for each voice
            st.markdown(f"""
            <div class="voice-card">
                <h4>{voice_name}</h4>
                <p><strong>Gender:</strong> {voice_data["gender"]} | <strong>Accent:</strong> {voice_data["accent"]}</p>
                <p>{voice_data["description"]}</p>
                <p><em>Sample:</em> "{voice_data["sample"]}"</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Voice comparison
    st.subheader("Compare Voices")
    st.markdown("Compare how different voices sound with the same text.")
    
    compare_text = st.text_input("Enter text for comparison:", "The quick brown fox jumps over the lazy dog.")
    compare_col1, compare_col2 = st.columns(2)
    
    with compare_col1:
        st.markdown("##### US Male vs US Female")
        st.info("Press 'Generate Speech' in the main tab to create comparisons")
        
    with compare_col2:
        st.markdown("##### US vs Indian Accent")
        st.info("Voice comparison samples will appear here")

with tab3:
    st.header("Speech History")
    
    # History filtering
    if st.session_state.history:
        filter_cols = st.columns(2)
        with filter_cols[0]:
            voice_history_filter = st.multiselect(
                "Filter by voice:", 
                options=list(set(item["voice"] for item in st.session_state.history)),
                default=list(set(item["voice"] for item in st.session_state.history))
            )
        with filter_cols[1]:
            sort_history = st.radio("Sort by:", ["Newest first", "Oldest first"], horizontal=True)
        
        # Display history items
        history_to_display = [
            item for item in st.session_state.history 
            if item["voice"] in voice_history_filter
        ]
        
        if sort_history == "Oldest first":
            history_to_display = history_to_display
        else:
            history_to_display = list(reversed(history_to_display))
        
        if not history_to_display:
            st.info("No items match your filter criteria")
        
        for i, item in enumerate(history_to_display):
            with st.expander(f"{item['timestamp']} - {item['voice']}"):
                st.markdown(f"**Text:** {item['text']}")
                st.markdown(f"**Voice:** {item['voice']}")
                st.markdown(f"**Settings:** Speed: {item['settings']['speed']}, Pitch: {item['settings']['pitch']}")
                
                if os.path.exists(item["file"]):
                    st.audio(item["file"], format="audio/wav")
                    col1, col2 = st.columns(2)
                    with col1:
                        with open(item["file"], "rb") as f:
                            st.download_button(
                                "⬇️ Download", 
                                f, 
                                file_name=f"tts_{i}.wav", 
                                mime="audio/wav",
                                key=f"download_{i}"
                            )
                    with col2:
                        if st.button("🔄 Regenerate", key=f"regen_{i}"):
                            st.session_state.regenerate_text = item["text"]
                            st.session_state.regenerate_voice = item["voice"]
                            st.rerun()
                else:
                    st.error("Audio file not found")
    else:
        st.info("Your speech history will appear here")
        st.markdown("""
        ### How to use history
        1. Generate speech in the Text to Speech tab
        2. All generated audio will be saved here
        3. You can replay, download, or regenerate past items
        """)

# Footer
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("Story Teller | Text to Speech | TTS")
with footer_col2:
    if in_docker:
        st.caption("TTS Demo Application (Docker Environment)")
    else:
        st.caption("TTS Demo Application (Local Environment)")
with footer_col3:
    st.markdown("<a href='#' style='text-decoration:none;'>Help</a> | <a href='#' style='text-decoration:none;'>About</a>", unsafe_allow_html=True)