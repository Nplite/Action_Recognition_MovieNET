import tqdm
import random
import pathlib
import collections
import json
import cv2
import einops
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers

# Data utilities

def list_files_per_class(zip_url):
    """List the files in each class of the dataset given the zip URL."""
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def get_class(fname):
    """Retrieve the name of the class given a filename."""
    return fname.split('_')[-3]

def get_files_per_class(files):
    """Retrieve the files that belong to each class."""
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def download_from_zip(zip_url, to_dir, file_names):
    """Download the contents of the zip file from the zip URL."""
    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(file_names):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn
            fn = pathlib.Path(fn).parts[-1]
            output_file = to_dir / class_name / fn
            unzipped_file.rename(output_file)

def split_class_lists(files_for_class, count):
    """Split the files belonging to a subset of data."""
    split_files = []
    remainder = {}
    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]
    return split_files, remainder

def download_ufc_101_subset(zip_url, num_classes, splits, download_dir):
    """Download a subset of the UFC101 dataset."""
    files = list_files_per_class(zip_url)
    for f in files:
        tokens = f.split('/')
        if len(tokens) <= 2:
            files.remove(f)
    
    files_for_class = get_files_per_class(files)
    classes = list(files_for_class.keys())[:num_classes]
    
    for cls in classes:
        new_files_for_class = files_for_class[cls]
        random.shuffle(new_files_for_class)
        files_for_class[cls] = new_files_for_class
    
    files_for_class = {x: files_for_class[x] for x in list(files_for_class)[:num_classes]}
    dirs = {}
    
    for split_name, split_count in splits.items():
        split_dir = download_dir / split_name
        split_files, files_for_class = split_class_lists(files_for_class, split_count)
        download_from_zip(zip_url, split_dir, split_files)
        dirs[split_name] = split_dir

    return dirs

# Frame utilities

def format_frames(frame, output_size):
    """Pad and resize an image from a video."""
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    """Creates frames from each video file."""
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))

    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

# Data generator

class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]
            yield video_frames, label

# Model

class Conv2Plus1D(layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)
        ])

    def call(self, x):
        return self.seq(x)

class ResidualMain(layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = tf.keras.Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

class Project(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.seq = tf.keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

def add_residual_block(input, filters, kernel_size):
    out = ResidualMain(filters, kernel_size)(input)
    res = input

    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])

class ResizeVideo(layers.Layer):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, '(b t) h w c -> b t h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(images, '(b t) h w c -> b t h w c', t=old_shape['t'])
        return videos

# Training and evaluation

# def build_model(input_shape, num_classes=10):
#     input = layers.Input(shape=input_shape[1:])
#     x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(input)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = ResizeVideo(112, 112)(x)

#     x = add_residual_block(x, 16, (3, 3, 3))
#     x = ResizeVideo(56, 56)(x)

#     x = add_residual_block(x, 32, (3, 3, 3))
#     x = ResizeVideo(28, 28)(x)

#     x = add_residual_block(x, 64, (3, 3, 3))
#     x = ResizeVideo(14, 14)(x)

#     x = add_residual_block(x, 128, (3, 3, 3))

#     x = layers.GlobalAveragePooling3D()(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(num_classes)(x)

#     model = tf.keras.Model(input, x)
#     return model

def build_model(input_shape, num_classes):
    input = tf.keras.Input(shape=input_shape)  # input_shape should be (num_frames, 224, 224, 3)
    
    # Reshape the input if needed
    x = tf.expand_dims(input, axis=1)  # Add a temporal dimension

    # First Conv2Plus1D layer
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)

    # Add more layers as necessary...

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)

    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='test')
    ax1.set_ylabel('Loss')
    max_loss = max(history.history['loss'] + history.history['val_loss'])
    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])

    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='test')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    plt.show()

# def run_experiment(model, train_dataset, val_dataset, epochs):
#     model.compile(
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         metrics=['accuracy']
#     )

#     history = model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=epochs
#     )
    
#     plot_history(history)
    
#     return history



def run_experiment(model, train_dataset, val_dataset, epochs, model_save_path="saved_model.h5", label_save_path="labels.json"):
    # Compile the model
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )
    
    # Plot training history
    plot_history(history)
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Assuming your dataset generator includes label mapping
    # Save the labels to a JSON file
    class_indices = train_dataset.class_indices  # Assuming this contains label mappings
    with open(label_save_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Labels saved to {label_save_path}")
    
    return history




if __name__ == "__main__":
    ZIP_URL = "https://storage.googleapis.com/thumos14_files/UCF101_videos.zip"  # Replace with actual dataset URL
    NUM_CLASSES = 2  # Adjust based on your dataset
    SPLITS = {"train": 70, "test": 30}  # Adjust as needed
    DOWNLOAD_DIR = pathlib.Path("./dataset")  # Specify the download directory
    
    # Download the dataset
    data_dirs = download_ufc_101_subset(ZIP_URL, NUM_CLASSES, SPLITS, DOWNLOAD_DIR)
    
    # Define Frame Generators
    train_generator = FrameGenerator(data_dirs['train'], n_frames=10, training=True)
    val_generator = FrameGenerator(data_dirs['test'], n_frames=10)
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_generator(train_generator, output_signature=(
        tf.TensorSpec(shape=(10, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ))
    
    val_dataset = tf.data.Dataset.from_generator(val_generator, output_signature=(
        tf.TensorSpec(shape=(10, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ))
    
    # Batch the datasets
    BATCH_SIZE = 4
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Build the model
    model = build_model(input_shape=(None, 224, 224, 3), num_classes=NUM_CLASSES)
    
    # Train the model
    # run_experiment(model, train_dataset, val_dataset, epochs=5)
    history = run_experiment(model, train_dataset, val_dataset, epochs=5, model_save_path="MovieNet.keras", label_save_path="MovieNet.json")

