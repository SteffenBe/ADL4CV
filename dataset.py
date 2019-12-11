import numpy as np
import torch
import itertools

from . import image_creation
from . import text_creation

all_shapes = ["square", "triangle"]
all_fill_colors = ["blue", "red", "green", "yellow", "purple", "cyan"]

all_dummy_classes = list(itertools.product(all_shapes, all_fill_colors))

def _assign_dummy_class(desc):
  """Infers one of the shape/color classes from a description."""

  for i, (shape, col) in enumerate(all_dummy_classes):
    if shape in desc and col in desc:
      return i
  raise ValueError('could not determine class of: ' + desc)


def generate_pairs(n_samples, image_resolution=64):
  """Generates a set of matching description/image pairs."""

  descriptions = np.ndarray((n_samples,), object)
  images = np.ndarray((n_samples, image_resolution, image_resolution, 3), np.float)

  background = image_creation.make_default_image('white')
  for i in range(n_samples):
    shape = np.random.choice(all_shapes)
    fill_color = np.random.choice(all_fill_colors)

    desc = text_creation.generate_single_description(shape, fill_color)
    img = image_creation.make_image(background, shape, fill_color, (image_resolution, image_resolution), super_sampling=4)

    descriptions[i] = desc
    images[i] = np.array(img) / 255
  return descriptions, images


def make_dataset(n_samples, vocab, random_seed):
  np.random.seed(random_seed)
  descriptions, images = generate_pairs(n_samples)

  word_sequences = torch.nn.utils.rnn.pad_sequence([
    torch.tensor(vocab.str_to_seq(d)) for d in descriptions
  ], batch_first=True)
  image_tensors = torch.tensor(images, dtype=torch.float32)
  dummy_labels = torch.from_numpy(np.fromiter((_assign_dummy_class(d) for d in descriptions), int))

  return torch.utils.data.TensorDataset(word_sequences, image_tensors, dummy_labels)


def _draw_positive_dummyclass(dummy_class, dummy_labels_np):
  indices = np.where(dummy_labels_np == dummy_class)[0]
  return np.random.choice(indices)

def _draw_negative_dummyclass(dummy_class, dummy_labels_np):
  indices = np.where(dummy_labels_np != dummy_class)[0]
  return np.random.choice(indices)


def make_triplet_dataset(n_samples, vocab, random_seed):
  np.random.seed(random_seed)
  descriptions, images = generate_pairs(n_samples)

  word_sequences = torch.nn.utils.rnn.pad_sequence([
    torch.tensor(vocab.str_to_seq(d)) for d in descriptions
  ], batch_first=True)
  image_tensors = torch.tensor(images, dtype=torch.float32)
  dummy_labels_np = np.fromiter((_assign_dummy_class(d) for d in descriptions), int)
  dummy_labels = torch.from_numpy(dummy_labels_np)

  word_sequences_positive = word_sequences.clone()
  word_sequences_negative = word_sequences.clone()
  image_tensors_positive = image_tensors.clone()
  image_tensors_negative = image_tensors.clone()

  for i, dummy_label in enumerate(dummy_labels_np):
    indice_pos = _draw_positive_dummyclass(dummy_label, dummy_labels_np)
    indice_neg = _draw_negative_dummyclass(dummy_label, dummy_labels_np)

    word_sequences_positive[i] = word_sequences[indice_pos]
    word_sequences_negative[i] = word_sequences[indice_neg]
    image_tensors_positive[i] = image_tensors[indice_pos]
    image_tensors_negative[i] = image_tensors[indice_neg]

  return torch.utils.data.TensorDataset(word_sequences_positive, image_tensors_positive,
                                        word_sequences, image_tensors, 
                                        word_sequences_negative, image_tensors_negative, 
                                        dummy_labels)
