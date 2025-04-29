# Real-Time Text to Speech (TTS) Project

This project implements a **Real-Time Text to Speech (TTS)** system leveraging **gRPC** for seamless communication between a backend **gRPC server** and a user-friendly **Streamlit UI**. The robust backend efficiently generates speech from user-provided text, while the intuitive frontend offers an interactive platform for users to experience real-time text-to-speech conversion.

## Project Structure

/project├── /server                     # gRPC server implementation│   ├── Dockerfile             # Dockerfile for building the gRPC server│   ├── requirements.txt       # Python dependencies for the server│   └── generated/            # Auto-generated files for gRPC├── /ui                        # Streamlit UI implementation│   ├── Dockerfile             # Dockerfile for building the UI container│   ├── requirements.txt       # Python dependencies for the UI│   └── ui.py                  # Streamlit UI script├── docker-compose.yml         # Docker Compose configuration for multi-container setup├── .dockerignore              # Files and directories to be ignored by Docker└── README.md                  # This file
## Prerequisites

* [Docker](https://www.docker.com/get-started)
* [Docker Compose](https://docs.docker.com/compose/install/)
* [Python 3.x](https://www.python.org/downloads/)

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Build the Docker containers:**

    Build both the UI and server containers using the following command:

    ```bash
    docker-compose up --build
    ```

    This command will automatically pull the necessary base images, build the Docker containers based on the provided `Dockerfile` in each service directory, and set up the internal network required for inter-container communication.

3.  **Run the containers:**

    To start both the gRPC server and the Streamlit UI, execute:

    ```bash
    docker-compose up
    ```

    Docker Compose will start the services defined in the `docker-compose.yml` file. The gRPC server will begin listening for requests, and the Streamlit UI will become accessible through your web browser.

## Accessing the Application

Once the Docker containers are successfully running, you can access the Streamlit UI via the following URLs:

* **Local URL:** `http://localhost:8501`
* **Network URL:** `http://<your-container-ip>:8501` (Replace `<your-container-ip>` with the IP address of the machine running the Docker containers if you are accessing it from another machine on the same network).

## API Overview

### gRPC Server

The backend gRPC server exposes a Text to Speech API endpoint that accepts text input and a description of the desired voice characteristics.

**Request Parameters:**

* `text`: `string` - The text content that needs to be converted into speech.
* `description`: `string` - A textual description of the desired voice, such as "A male speaker with a deep voice," "A cheerful female voice," or "A robotic voice with a fast speaking rate." This parameter allows for some control over the generated speech characteristics.

**Response:**

* The API returns the generated audio data in `.wav` format as a stream of bytes.

### Streamlit UI

The frontend Streamlit UI provides a simple and intuitive interface for users to interact with the Text to Speech system.

**User Input Fields:**

* **Text:** A text input field where users can type or paste the text they wish to convert to speech.
* **Speaker Description:** A text input field where users can provide a description of the desired speaker's voice (e.g., tone, speed, gender).

**Functionality:**

* **"Generate Speech" Button:** Upon clicking this button, the UI takes the text and speaker description provided by the user and sends a gRPC request to the backend server.
* **Audio Playback:** Once the server processes the request and sends back the audio data, the UI automatically plays the generated speech directly in the user's browser.

## Troubleshooting

Here are some common issues you might encounter and potential solutions:

**Connection Refused Error:**

* **Symptom:** The Streamlit UI might display an error indicating that it cannot connect to the gRPC server.
* **Possible Cause:** One or both of the Docker containers (server and UI) might not be running, or the gRPC channel in the `ui.py` script is configured with an incorrect server address.
* **Solution:**
    * Ensure both the `server` and `ui` services are listed as running when you execute `docker-compose ps`. If any are stopped, try restarting them with `docker-compose restart <service_name>`.
    * Verify that the gRPC channel in your `ui/ui.py` script is correctly set to the internal Docker network address of the server: `server:50051`. Docker Compose automatically resolves service names to their respective container IPs within the network.

**Streamlit App Not Showing Up:**

* **Symptom:** You cannot access the Streamlit UI at `http://localhost:8501` or the network URL.
* **Possible Causes:**
    * The `ui` container might not have started correctly.
    * Port `8501` on your host machine might be blocked by a firewall.
    * Another process on your machine might be using port `8501`.
* **Solutions:**
    * Check the logs of the `ui` container using `docker-compose logs ui` for any error messages that might indicate why the application failed to start.
    * Ensure that your firewall is configured to allow incoming connections on port `8501`.
    * Use the command `netstat -tuln | grep 8501` (on Linux/macOS) or `Get-NetTCPConnection -LocalPort 8501` (on PowerShell) to check if any other process is listening on port `8501`. If so, you might need to stop that process or configure the Streamlit app to run on a different port (which would require modifying the `Dockerfile` and potentially the `docker-compose.yml`).

By following these instructions, you should be able to set up and run the Real-Time Text to Speech project successfully. Enjoy experimenting with real-time text-to-speech conversion!
