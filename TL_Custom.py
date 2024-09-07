
import tqdm
import random
import pathlib
import itertools
import collections
import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
from pathlib import Path


def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        # start = random.randint(0, max_start + 1)
        start = random.randint(0, int(max_start) + 1)

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


def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes
    )
    model.build([batch_size, num_frames, resolution, resolution, 3])
    return model




def main():
    batch_size = 8
    num_frames = 8
    num_epochs = 2
    resolution = 224

    subset_paths = {
        'train': Path('dataset/train'),
        'test': Path('dataset/test')
    }

    output_signature = (
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int16)
    )

    train_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths['train'], num_frames, training=True),
        output_signature=output_signature
    ).batch(batch_size)

    test_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_paths['test'], num_frames),
        output_signature=output_signature
    ).batch(batch_size)

    for frames, labels in train_ds.take(10):
        print(labels)

    gru = layers.GRU(units=4, return_sequences=True, return_state=True)
    inputs = tf.random.normal(shape=[1, 10, 8])
    result, state = gru(inputs)
    first_half, state = gru(inputs[:, :5, :])
    second_half, _ = gru(inputs[:, 5:, :], initial_state=state)

    print(np.allclose(result[:, :5, :], first_half))
    print(np.allclose(result[:, 5:, :], second_half))

    model_id = 'a0'
    backbone = movinet.Movinet(model_id=model_id)
    backbone.trainable = False

    model = build_classifier(batch_size, num_frames, resolution, backbone, num_classes=10)
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    results = model.fit(train_ds, validation_data=test_ds, epochs=num_epochs, validation_freq=1, verbose=1)
    model.evaluate(test_ds, return_dict=True)

    # model.save("Movinet_Model.h5")
    model.save("Movinet_Model.keras")
    model.save("Movinet_Model.h5")
    print("Model saved successfully...", model.summary())
    

 


if __name__ == "__main__":
    main()








