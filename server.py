import socket
import cv2
import numpy as np
from enlighten_inference import EnlightenOnnxModel
import concurrent.futures
import socket

server = socket.socket()
server.bind(('localhost', 12345))

print("Listening on port 12345...")

while True:
  client, address = server.accept()
  print(f"Connection from {address} received!")

  # Receive and send data
  data = client.recv(1024)
  client.send(data)

  client.close()

# Create a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 12345))
server_socket.listen(1)

# Initialize the EnlightenGAN model
enlighten_model = EnlightenOnnxModel()

# Create a thread pool with a maximum of 4 worker threads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def handle_client(client_socket):
    try:
        batch = []  # Initialize an empty batch for frame processing
        print("Client connected")  # Print when a client connects
        while True:
            # Receive a frame from the client
            frame_data = client_socket.recv(1024)
            if not frame_data:
                break

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # Append the frame to the batch for processing
            batch.append(frame)

            # Process the batch if it reaches a certain size (e.g., 4 frames)
            if len(batch) >= 4:
                enhanced_batch = [enlighten_model.predict(frame) for frame in batch]
                print(f"Enhanced {len(batch)} frames")  # Print when frames are enhanced

                # Send the enhanced frames back to the client
                enhanced_frame_data = [cv2.imencode('.jpg', frame)[1].tostring() for frame in enhanced_batch]
                for data in enhanced_frame_data:
                    client_socket.send(data)
                print(f"Sent {len(batch)} enhanced frames back to the client")  # Print when enhanced frames are sent

                # Clear the batch
                batch.clear()

    except Exception as e:
        print(f"Client handling error: {e}")  # Print any errors during client handling
    finally:
        client_socket.close()

# Accept client connections and handle them in a thread
print("Server is listening for connections...")
while True:
    (client_socket, _) = server_socket.accept()
    executor.submit(handle_client, client_socket)

# Close the socket when done
server_socket.close()
