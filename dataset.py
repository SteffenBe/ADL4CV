import numpy as np
import torch
import itertools

from . import image_creation
from . import text_creation


from torch.utils.data import TensorDataset

all_shapes = ["square", "triangle"]
all_fill_colors = ["blue", "red", "green", "yellow", "purple", "cyan"]

all_dummy_classes = list(itertools.product(all_shapes, all_fill_colors))

def _assign_dummy_class(desc):
  """Infers one of the shape/color classes from a description."""

  for i, (shape, col) in enumerate(all_dummy_classes):
    if shape in desc and col in desc:
      return i
  raise ValueError('could not determine class of: ' + desc)


def generate_pairs(n_samples, image_resolution=64, blacklist=[]):
  """Generates a set of matching description/image pairs."""

  # Validate tuple order in blacklist to prevent passing in wrong order by accident.
  if len(blacklist) > 0 and not isinstance(blacklist[0], tuple):
    raise ValueError("Blacklist should be a list of tuples!")
  if len(blacklist) > 0 and (blacklist[0][0] in all_fill_colors or blacklist[0][1] in all_shapes):
    raise ValueError("Blacklist should consist of (shape, color) tuples, not (color, shape) tuples!")

  descriptions = np.ndarray((n_samples,), object)
  images = np.ndarray((n_samples, image_resolution, image_resolution, 3), np.float)

  background = image_creation.make_default_image('white')
  for i in range(n_samples):
    # Pick a non-blacklisted shape/color combination.
    shape = np.random.choice(all_shapes)
    fill_color = np.random.choice(all_fill_colors)
    while (shape, fill_color) in blacklist:
      shape = np.random.choice(all_shapes)
      fill_color = np.random.choice(all_fill_colors)

    desc = text_creation.generate_single_description(shape, fill_color)
    img = image_creation.make_image(background, shape, fill_color, (image_resolution, image_resolution), super_sampling=4)

    descriptions[i] = desc
    images[i] = np.array(img) / 255
  return descriptions, images


def make_dataset(n_samples, vocab, random_seed, blacklist=[]):
  np.random.seed(random_seed)
  descriptions, images = generate_pairs(n_samples, blacklist=blacklist)

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

def make_triplet_dataset_original(n_samples, vocab, random_seed, blacklist=[]):
  np.random.seed(random_seed)
  descriptions, images = generate_pairs(n_samples, blacklist=blacklist)

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

def make_triplet_dataset(n_samples, vocab, random_seed, blacklist=[], train=True):
  np.random.seed(random_seed)
  descriptions, images = generate_pairs(n_samples, blacklist=blacklist)

  word_sequences = torch.nn.utils.rnn.pad_sequence([
    torch.tensor(vocab.str_to_seq(d)) for d in descriptions
  ], batch_first=True)
  image_tensors = torch.tensor(images, dtype=torch.float32)
  dummy_labels_np = np.fromiter((_assign_dummy_class(d) for d in descriptions), int)
  dummy_labels = torch.from_numpy(dummy_labels_np)

  return TripletDataset(word_sequences, image_tensors, dummy_labels, train=train)

class TripletDataset(TensorDataset):
  """
  Train: For each sample (anchor) randomly chooses a positive and negative samples
  Test: Creates fixed triplets for testing
  """
  def __init__(self, *tensors, train=True):
    super(TripletDataset, self).__init__(*tensors)
    self.train = train

    self.labels = self.tensors[-1]
    self.labels_set = set(self.labels.numpy())
    self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                             for label in self.labels_set}

    if not self.train:

      random_state = np.random.RandomState(29)

      triplets = [[i,
                   random_state.choice(self.label_to_indices[self.labels[i].item()]),
                   random_state.choice(self.label_to_indices[
                                         np.random.choice(
                                           list(self.labels_set - set([self.labels[i].item()]))
                                         )
                                       ])
                   ]
                  for i in range(self.tensors[0].size(0))]

      self.test_triplets = triplets

  def __getitem__(self, index):
    if self.train:
      text_anchor, image_anchor, label = self.tensors[0][index], self.tensors[1][index], self.tensors[2][index]
      #img1, label1 = self.train_data[index], self.train_labels[index].item()
      positive_index = index
      while positive_index == index:
        positive_index = np.random.choice(self.label_to_indices[label])
      negative_label = np.random.choice(list(self.labels_set - set([label])))
      negative_index = np.random.choice(self.label_to_indices[negative_label])
      text_positive = self.tensors[0][positive_index]
      image_positive = self.tensors[1][positive_index]
      text_negative = self.tensors[0][negative_index]
      image_negative = self.tensors[1][negative_index]

    else:
      text_anchor = self.tensors[0][self.test_triplets[index][0]]
      image_anchor = self.tensors[1][self.test_triplets[index][0]]
      text_positive = self.tensors[0][self.test_triplets[index][1]]
      image_positive = self.tensors[1][self.test_triplets[index][1]]
      text_negative = self.tensors[0][self.test_triplets[index][2]]
      image_negative = self.tensors[1][self.test_triplets[index][2]]

    return tuple([text_positive, image_positive, text_anchor, image_anchor, text_negative, image_negative, self.tensors[2][index]])

  def __len__(self):
    return self.tensors[0].size(0)