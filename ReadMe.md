# TTS gRPC Microservice

This project provides a gRPC microservice to generate text-to-speech audio using Parler-TTS.

## How to Run

1. Install requirements
    ```
    pip install -r requirements.txt
    ```

2. Compile proto
    ```
    python -m grpc_tools.protoc -I=server --python_out=generated --grpc_python_out=generated server/massanger.proto
    ```

3. Start server
    ```
    python server/server.py
    ```

4. Run client
    ```
    python client/client.py
    ```

## Folder Structure
- `server/` - gRPC server and model code
- `client/` - Client to interact with server
- `generated/` - Generated code from proto
