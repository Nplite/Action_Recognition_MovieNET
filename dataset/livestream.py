import cv2
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib as mpl


# Constants
IMAGE_SIZE = (224, 224)
FPS = 8.0  # Desired Frames Per Second for model inference

# Update matplotlib params
mpl.rcParams.update({'font.size': 10})

def load_labels():
    """Load the Kinetics 600 labels from a given URL."""
    labels_url = 'https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
    labels_path = tf.keras.utils.get_file(fname='labels.txt', origin=labels_url)
    lines = pathlib.Path(labels_path).read_text().splitlines()
    return np.array([line.strip() for line in lines])



KINETICS_600_LABELS = load_labels()

def preprocess_frame(frame, image_size=IMAGE_SIZE):
    """Preprocess a single frame for model input."""
    frame = cv2.resize(frame, image_size)
    frame = tf.convert_to_tensor(frame, dtype=tf.float32) / 255.0
    return frame

def load_movinet_model(model_id='a2', mode='base', version='3'):
    """Load a MoViNet model from TensorFlow Hub."""
    hub_url = f'https://tfhub.dev/tensorflow/movinet/{model_id}/{mode}/kinetics-600/classification/{version}'
    return hub.load(hub_url)

def get_top_k(probs, k=5, label_map=None):
    """Get the top-k predictions from the probability tensor."""
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))


def run_streaming_inference(frame, model, state):
    """Run streaming inference on a frame using a model."""
    inputs = state
    inputs['image'] = frame[tf.newaxis, tf.newaxis, ...]
    
    # Run model inference
    result, state = model(inputs)

    # Since the result is a tensor, no need to index it
    logits = result  # Directly assign the result to logits
    
    probabilities = tf.nn.softmax(logits, axis=-1)
    return probabilities, state


def main():
    # Load labels
    kinetics_labels = load_labels()

    # Load MoViNet model for streaming
    model_stream = load_movinet_model(mode='stream')
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize state for streaming model
    initial_state = model_stream.init_states([1, 1, *IMAGE_SIZE, 3])

    print("Starting webcam live stream... Press 'q' to exit.")
    
    state = initial_state
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Run inference on the current frame
        probabilities, state = run_streaming_inference(processed_frame, model_stream, state)

        # Get top-k predictions
        top_k_labels = get_top_k(probabilities[0], label_map=KINETICS_600_LABELS)
        
        # Display predictions on the frame
        for i, (label, prob) in enumerate(top_k_labels):
            text = f"{label}: {prob:.2f}"
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow("Webcam Live Action Recognition", frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
