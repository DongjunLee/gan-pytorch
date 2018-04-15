
import os
from io import BytesIO
import numpy as np
import re
import scipy.misc
import tensorflow as tf
import torch



def load_saved_model(path, model, optimizer):
    latest_path = find_latest(path)
    if latest_path is None:
        return 0, model, optimizer

    checkpoint = torch.load(latest_path)

    step_count = checkpoint['step_count']
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"Load checkpoints...! {latest_path}")
    return step_count, model, optimizer


def find_latest(find_path):
    sorted_path = get_sorted_path(find_path)
    if len(sorted_path) == 0:
        return None

    return sorted_path[-1]


def save_checkpoint(step, path, model, optimizer, max_to_keep=10):
    sorted_path = get_sorted_path(path)
    for i in range(len(sorted_path) - max_to_keep):
        os.remove(sorted_path[i])

    full_path = path + f"-{step}.pkl"
    torch.save({
        "step_count": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, full_path)
    print(f"Save checkpoints...! {full_path}")


def get_sorted_path(find_path):
    dir_path = os.path.dirname(find_path)
    base_name = os.path.basename(find_path)

    paths = []
    for root, dirs, files in os.walk(dir_path):
        for f_name in files:
            if f_name.startswith(base_name) and f_name.endswith(".pkl"):
                paths.append(os.path.join(root, f_name))

    return sorted(paths, key=lambda x: int(re.findall("\d+", os.path.basename(x))[0]))


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class TensorBoard:

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag=f"{tag}/{i}", image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
