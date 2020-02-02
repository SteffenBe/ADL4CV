import numpy as np
import torch
import itertools
from torch.utils.data import TensorDataset
from typing import List, Tuple

# Optional module for showing progress bars.
try: from tqdm import tqdm
except ModuleNotFoundError: tqdm = lambda x: x

from . import image_creation
from . import text_creation


all_shapes = ["square", "triangle"]
all_fill_colors = ["blue", "red", "green", "yellow", "purple", "cyan"]

all_dummy_classes = list(itertools.product(all_shapes, all_fill_colors))


def choose_attrs(blacklist: List[Tuple[str, str]]) -> Tuple[str, str]:
  """Picks a non-blacklisted shape/color combination."""

  shape = np.random.choice(all_shapes)
  fill_color = np.random.choice(all_fill_colors)
  while (shape, fill_color) in blacklist:
    shape = np.random.choice(all_shapes)
    fill_color = np.random.choice(all_fill_colors)
  return shape, fill_color


def generate_samples(n_samples, image_resolution=64, blacklist=[]):
  """Returns a generator that yields n_samples times individual tuples of the shape:
      (description: str,
        image: numpy array with shape (R, R, C),
        label: int)"""

  # Validate tuple order in blacklist to prevent passing in wrong order by accident.
  if len(blacklist) > 0 and not isinstance(blacklist[0], tuple):
    raise ValueError("Blacklist should be a list of tuples!")
  if len(blacklist) > 0 and (blacklist[0][0] in all_fill_colors or blacklist[0][1] in all_shapes):
    raise ValueError("Blacklist should consist of (shape, color) tuples, not (color, shape) tuples!")

  background = image_creation.make_default_image('white')
  for i in range(n_samples):
    shape, fill_color = choose_attrs(blacklist)
    label = all_dummy_classes.index((shape, fill_color))
    desc = text_creation.generate_single_description(shape, fill_color)
    img = image_creation.make_image(background, shape, fill_color, (image_resolution, image_resolution), super_sampling=2)
    img_np = np.array(img, dtype=np.float32)
    img_np /= 255.0

    yield desc, img_np, label


def make_samples_numpy(n_samples, image_resolution=64, blacklist=[]):
  """Like generate_samples, but actually collects all results into numpy arrays.
  Returns numpy arrays: (
    descriptions: (N,) string (^= object),
    images: (N, R, R, C) float,
    labels: (N,) int
  )"""

  descriptions = np.ndarray((n_samples,), object)
  images = np.ndarray((n_samples, image_resolution, image_resolution, 3), np.float)
  labels = np.ndarray((n_samples,), np.int)
  
  generator = generate_samples(n_samples, image_resolution, blacklist=blacklist)
  if n_samples >= 2000:
    generator = tqdm(generator, total=n_samples)
  for i, (description, image, label) in enumerate(generator):
    descriptions[i] = description
    images[i] = image
    labels[i] = label
  return descriptions, images, labels


def make_samples_tensors(n_samples, vocab, image_resolution=64, blacklist=[]):
  """Generates samples, applies vocabulary mapping and collects them into tensors.
  Returns: (
    word_sequences: int tensor shape (N, L) where L is word count of longest description,
    image_tensors: float tensor shape (N, R, R, C),
    labels: int tensor shape (N,)
  )"""
  # Allocate full tensors exactly like we will store them in the dataset.
  # Except for descriptions, which need to be padded after knowing all of them.
  unpadded_word_sequences = [None] * n_samples
  image_tensors = torch.empty(n_samples, image_resolution, image_resolution, 3, dtype=torch.float32)
  labels = torch.empty(n_samples, dtype=torch.int)
  
  # Fill tensors.
  generator = generate_samples(n_samples, image_resolution, blacklist=blacklist)
  if n_samples >= 2000:
    generator = tqdm(generator, total=n_samples)
  for i, (description, image, label) in enumerate(generator):
    seq = vocab.str_to_seq(description)
    unpadded_word_sequences[i] = torch.tensor(seq, dtype=torch.long)
    image_tensors[i] = torch.as_tensor(image)
    labels[i] = label

  word_sequences = torch.nn.utils.rnn.pad_sequence(unpadded_word_sequences, batch_first=True)
  return word_sequences, image_tensors, labels


def make_dataset(n_samples, vocab, random_seed, blacklist=[]):
  np.random.seed(random_seed)
  word_sequences, image_tensors, labels = make_samples_tensors(n_samples, vocab, blacklist=blacklist)
  return torch.utils.data.TensorDataset(word_sequences, image_tensors, labels)


def make_triplet_dataset(n_samples, vocab, random_seed, blacklist=[], train=True):
  np.random.seed(random_seed)
  word_sequences, image_tensors, labels = make_samples_tensors(n_samples, vocab, blacklist=blacklist)
  return TripletDataset(word_sequences, image_tensors, labels, train=train)


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
      text_anchor, image_anchor, label = self.tensors[0][index], self.tensors[1][index], self.tensors[2][index].item()
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



def make_modifier_dataset(n_samples, vocab, text_encoder, random_seed=None, blacklist=[]):
  if random_seed is not None:
    np.random.seed(random_seed)
  unpadded_sequences_in = [None] * n_samples
  unpadded_sequences_mod = [None] * n_samples
  unpadded_sequences_out = [None] * n_samples
  
  for i in tqdm(range(n_samples)):
    shape, fill_color = choose_attrs(blacklist)
    desc_tpl = text_creation.choose_baked_description_template()
    # Generate sample for in_embedding.
    desc_in = text_creation.generate_single_description(shape, fill_color, desc_tpl)
    unpadded_sequences_in[i] = torch.tensor(vocab.str_to_seq(desc_in), dtype=torch.long)

    # Pick either a change in shape or color.
    modifications = {}
    if np.random.randint(2) == 0:
      shape = np.random.choice(list(set(all_shapes) - set([shape])))
      modifications['new_shape'] = shape
    else:
      fill_color = np.random.choice(list(set(all_fill_colors) - set([fill_color])))
      modifications['new_color'] = fill_color

    # Generate sample for out_embedding.
    desc_out = text_creation.generate_single_description(shape, fill_color, desc_tpl)
    unpadded_sequences_out[i] = torch.tensor(vocab.str_to_seq(desc_out), dtype=torch.long)
    
    # Generate modification text sample.
    mod_str = text_creation.generate_single_modification(**modifications)
    unpadded_sequences_mod[i] = torch.tensor(vocab.str_to_seq(mod_str), dtype=torch.long)

  # Pad to equal length.
  sequences_in = torch.nn.utils.rnn.pad_sequence(unpadded_sequences_in, batch_first=True)
  sequences_out = torch.nn.utils.rnn.pad_sequence(unpadded_sequences_out, batch_first=True)
  sequences_mod = torch.nn.utils.rnn.pad_sequence(unpadded_sequences_mod, batch_first=True)

  device = next(joint_model.parameters()).device
  in_embeddings_tensor = text_encoder(sequences_in.to(device)).detach().cpu()
  out_embeddings_tensor = text_encoder(sequences_out.to(device)).detach().cpu()
  return torch.utils.data.TensorDataset(in_embeddings_tensor, sequences_mod, out_embeddings_tensor)
