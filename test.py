# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# from official.projects.movinet.modeling import movinet
# from official.projects.movinet.modeling import movinet_model

# def format_frames(frame, output_size):
#     frame = frame[:, :, [2, 1, 0]]
#     frame = tf.image.convert_image_dtype(frame, tf.float32)
#     frame = tf.image.resize_with_pad(frame, *output_size)
#     return frame

# def build_classifier(backbone, num_classes):
#     model = movinet_model.MovinetClassifier(
#         backbone=backbone,
#         num_classes=num_classes
#     )
#     return model

# def main():
#     # Parameters
#     batch_size = 1
#     num_frames = 8
#     num_classes = 10  # Adjust this based on your dataset
#     resolution = 224
#     model_id = 'a0'  # Adjust this if you used a different MoViNet variant

#     # Build the model
#     backbone = movinet.Movinet(model_id=model_id)
#     backbone.trainable = False
#     model = build_classifier(backbone, num_classes)
#     model.build([batch_size, num_frames, resolution, resolution, 3])

#     # Load the trained weights
#     model.load_weights("Movinet_Model.keras")

#     # Get the list of class names (adjust this based on your dataset)
#     class_names = ["CrikectBowling", "JavelinThrow", "ThrowDiscuss" ]

#     # Set up video capture
#     cap = cv2.VideoCapture(0)

#     # Frame buffer
#     frame_buffer = deque(maxlen=num_frames)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocess the frame
#         processed_frame = format_frames(frame, (resolution, resolution))
#         frame_buffer.append(processed_frame)

#         if len(frame_buffer) == num_frames:
#             # Prepare input for the model
#             model_input = tf.stack(list(frame_buffer))
#             model_input = tf.expand_dims(model_input, axis=0)

#             # Make prediction
#             prediction = model(model_input)
#             predicted_class = class_names[np.argmax(prediction[0])]
#             confidence = tf.nn.softmax(prediction[0])[np.argmax(prediction[0])].numpy()

#             # Display the result on the frame
#             cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Display the frame
#         cv2.imshow('MoViNet Webcam Test', frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

@tf.function
def predict(model, input_frames):
    return model(input_frames, training=False)

def build_classifier(backbone, num_classes):
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes
    )
    return model

def main():
    # Parameters
    batch_size = 1
    num_frames = 8
    num_classes = 10  # Adjust this based on your dataset
    resolution = 224
    model_id = 'a0'  # Adjust this if you used a different MoViNet variant

    # Build the model
    backbone = movinet.Movinet(model_id=model_id)
    backbone.trainable = False
    model = build_classifier(backbone, num_classes)
    model.build([batch_size, num_frames, resolution, resolution, 3])

    # Load the trained weights
    model.load_weights("Movinet_Model.keras")

    # Get the list of class names (adjust this based on your dataset)
    class_names = ["CrikectBowling", "JavelinThrow", "ThrowDiscuss" ]

    # Set up video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Frame buffer
    frame_buffer = deque(maxlen=num_frames)

    # For FPS calculation
    prev_time = 0
    
    with tf.device('/GPU:0'):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            current_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_time - prev_time)
            prev_time = current_time

            # Preprocess the frame
            processed_frame = format_frames(frame, (resolution, resolution))
            frame_buffer.append(processed_frame)

            if len(frame_buffer) == num_frames:
                # Prepare input for the model
                model_input = tf.stack(list(frame_buffer))
                model_input = tf.expand_dims(model_input, axis=0)

                # Make prediction
                prediction = predict(model, model_input)
                predicted_class = class_names[np.argmax(prediction[0])]
                confidence = tf.nn.softmax(prediction[0])[np.argmax(prediction[0])].numpy()

                # Display the result on the frame
                cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('MoViNet Webcam Test', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


