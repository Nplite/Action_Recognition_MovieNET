import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import tqdm

# Constants
# LABELS_URL = 'https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
JUMPINGJACK_URL = 'dance.gif'
IMAGE_SIZE = (224, 224)



# Update matplotlib params
mpl.rcParams.update({'font.size': 10})

def load_labels():

    """Load the Kinetics 600 labels from a given URL."""
    labels_url = 'https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
    labels_path = tf.keras.utils.get_file(fname='labels.txt', origin=labels_url)
    lines = pathlib.Path(labels_path).read_text().splitlines()
    return np.array([line.strip() for line in lines])

KINETICS_600_LABELS = load_labels()

def load_gif(file_path, image_size=IMAGE_SIZE):
    """Loads a GIF file into a TF tensor."""
    raw = tf.io.read_file(file_path)
    video = tf.io.decode_gif(raw)
    video = tf.image.resize(video, image_size)
    video = tf.cast(video, tf.float32) / 255.0
    return video

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

def run_streaming_inference(video, model, initial_state):
    """Run streaming inference on a video using a model."""
    state = initial_state.copy()
    all_logits = []

    for n in tqdm.tqdm(range(len(video))):
        inputs = state
        inputs['image'] = video[tf.newaxis, n:n+1, ...]

        # Run model inference
        result, state = model(inputs)

        # Inspect the structure of `result` to understand what it contains
        if n == 0:
            print("Result structure:", result)

        # Append logits (modify this depending on the result structure)
        if isinstance(result, dict):
            all_logits.append(result['classifier_head'])  # Use the correct key
        else:
            all_logits.append(result)  # If result is directly the tensor

    return tf.nn.softmax(tf.concat(all_logits, axis=0), axis=-1)



#@title
# Get top_k labels and probabilities predicted using MoViNets streaming model
def get_top_k_streaming_labels(probs, k=5, label_map = KINETICS_600_LABELS):
  """Returns the top-k labels over an entire video sequence.

  Args:
    probs: probability tensor of shape (num_frames, num_classes) that represents
      the probability of each class on each frame.
    k: the number of top predictions to select.
    label_map: a list of labels to map logit indices to label strings.

  Returns:
    a tuple of the top-k probabilities, labels, and logit indices
  """
  top_categories_last = tf.argsort(probs, -1, 'DESCENDING')[-1, :1]
  # Sort predictions to find top_k
  categories = tf.argsort(probs, -1, 'DESCENDING')[:, :k]
  categories = tf.reshape(categories, [-1])

  counts = sorted([
      (i.numpy(), tf.reduce_sum(tf.cast(categories == i, tf.int32)).numpy())
      for i in tf.unique(categories)[0]
  ], key=lambda x: x[1], reverse=True)

  top_probs_idx = tf.constant([i for i, _ in counts[:k]])
  top_probs_idx = tf.concat([top_categories_last, top_probs_idx], 0)
  # find unique indices of categories
  top_probs_idx = tf.unique(top_probs_idx)[0][:k+1]
  # top_k probabilities of the predictions
  top_probs = tf.gather(probs, top_probs_idx, axis=-1)
  top_probs = tf.transpose(top_probs, perm=(1, 0))
  # collect the labels of top_k predictions
  top_labels = tf.gather(label_map, top_probs_idx, axis=0)
  # decode the top_k labels
  top_labels = [label.decode('utf8') for label in top_labels.numpy()]

  return top_probs, top_labels, top_probs_idx

# Plot top_k predictions at a given time step
def plot_streaming_top_preds_at_step(
    top_probs,
    top_labels,
    step=None,
    image=None,
    legend_loc='lower left',
    duration_seconds=10,
    figure_height=500,
    playhead_scale=0.8,
    grid_alpha=0.3):
  """Generates a plot of the top video model predictions at a given time step.

  Args:
    top_probs: a tensor of shape (k, num_frames) representing the top-k
      probabilities over all frames.
    top_labels: a list of length k that represents the top-k label strings.
    step: the current time step in the range [0, num_frames].
    image: the image frame to display at the current time step.
    legend_loc: the placement location of the legend.
    duration_seconds: the total duration of the video.
    figure_height: the output figure height.
    playhead_scale: scale value for the playhead.
    grid_alpha: alpha value for the gridlines.

  Returns:
    A tuple of the output numpy image, figure, and axes.
  """
  # find number of top_k labels and frames in the video
  num_labels, num_frames = top_probs.shape
  if step is None:
    step = num_frames
  # Visualize frames and top_k probabilities of streaming video
  fig = plt.figure(figsize=(6.5, 7), dpi=300)
  gs = mpl.gridspec.GridSpec(8, 1)
  ax2 = plt.subplot(gs[:-3, :])
  ax = plt.subplot(gs[-3:, :])
  # display the frame
  if image is not None:
    ax2.imshow(image, interpolation='nearest')
    ax2.axis('off')
  # x-axis (frame number)
  preview_line_x = tf.linspace(0., duration_seconds, num_frames)
  # y-axis (top_k probabilities)
  preview_line_y = top_probs

  line_x = preview_line_x[:step+1]
  line_y = preview_line_y[:, :step+1]

  for i in range(num_labels):
    ax.plot(preview_line_x, preview_line_y[i], label=None, linewidth='1.5',
            linestyle=':', color='gray')
    ax.plot(line_x, line_y[i], label=top_labels[i], linewidth='2.0')


  ax.grid(which='major', linestyle=':', linewidth='1.0', alpha=grid_alpha)
  ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=grid_alpha)

  min_height = tf.reduce_min(top_probs) * playhead_scale
  max_height = tf.reduce_max(top_probs)
  ax.vlines(preview_line_x[step], min_height, max_height, colors='red')
  ax.scatter(preview_line_x[step], max_height, color='red')

  ax.legend(loc=legend_loc)

  plt.xlim(0, duration_seconds)
  plt.ylabel('Probability')
  plt.xlabel('Time (s)')
  plt.yscale('log')

  fig.tight_layout()
  fig.canvas.draw()

  data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()

  figure_width = int(figure_height * data.shape[1] / data.shape[0])
  image = PIL.Image.fromarray(data).resize([figure_width, figure_height])
  image = np.array(image)

  return image

def plot_streaming_top_preds(probs, video, top_k=5, video_fps=25.0, figure_height=500):
    """Generate a plot of the top predictions over a streaming video."""
    top_probs, top_labels, _ = get_top_k_streaming_labels(probs, k=top_k)
    images = []

    for i in tqdm.trange(len(video)):
        image = plot_streaming_top_preds_at_step(
            top_probs=top_probs,
            top_labels=top_labels,
            step=i,
            image=video[i],
            duration_seconds=len(video) / video_fps,
            figure_height=figure_height
        )
        images.append(image)

    return np.array(images)

def main():
    # Load labels
    kinetics_labels = load_labels()

    # Load GIF
    jumpingjack_path = tf.keras.utils.get_file(
        fname='jumpingjack.gif',
        origin=JUMPINGJACK_URL,
        cache_dir='.',
        cache_subdir='.'
    )
    jumpingjack = load_gif(jumpingjack_path)

    # Load MoViNet model
    model = load_movinet_model()
    sig = model.signatures['serving_default']
    sig(image=jumpingjack[tf.newaxis, :1])

    # Get logits and predictions
    logits = sig(image=jumpingjack[tf.newaxis, ...])['classifier_head'][0]
    probs = tf.nn.softmax(logits, axis=-1)
    
    print("Top-k predictions and their probabilities:")
    for label, p in get_top_k(probs, label_map=kinetics_labels):
        print(f'{label:20s}: {p:.3f}')

    # Load streaming MoViNet model and initialize state
    model_stream = load_movinet_model(mode='stream')
    initial_state = model_stream.init_states(jumpingjack[tf.newaxis, ...].shape)

    # Run streaming inference on the full video
    probabilities = run_streaming_inference(jumpingjack, model_stream, initial_state)

    print("\nTop-k predictions from streaming model:")
    for label, p in get_top_k(probabilities[-1], label_map=kinetics_labels):
        print(f'{label:20s}: {p:.3f}')

    # Plot and show predictions over the video
    plot_video = plot_streaming_top_preds(probabilities, jumpingjack, video_fps=8.0)
    media.show_video(plot_video, fps=3)

if __name__ == "__main__":
    main()
