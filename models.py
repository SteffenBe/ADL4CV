import torch
from torch import nn
from datetime import datetime

def count_params(model):
  """Returns the total number of trainable parameters in a torch module."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(label, model, verbose=True):
  f = 'joint_model--{0}--{1}.pt'.format(count_params(joint_model), datetime.utcnow().strftime('%Y-%m-%dT%H-%M'))
  torch.save(joint_model.state_dict(), f)
  if verbose:
    print('Model saved at "{}"'.format(f))


class TextEncoder(nn.Module):
  """The text encoder takes a sequence of word indices and outputs
  the joint sequence embedding (J)."""

  def __init__(self, vocab_size, word_embedding_size, out_size):
    super().__init__()
    self.word_embedding = nn.Embedding(vocab_size, word_embedding_size, padding_idx=0)
    self.rnn = nn.LSTM(word_embedding_size, out_size, batch_first=True)
  
  def forward(self, x):
    x = self.word_embedding(x)
    x, (hidden, cell) = self.rnn(x)
    # Return last output of the sequence (final embedding).
    return x[:, -1, :]


class ImageEncoder(nn.Module):
  """The image encoder takes an input image and outputs the joint image embedding (J).
  
  Input shape: (N, in_width, in_height, in_channels)
  Output shape: (N, out_size)
  """

  def __init__(self, in_width, in_height, in_channels, out_size):
    super().__init__()
    pooling_factor = 8  # 2**num_pool_layers
    num_last_conv_channels = 64
    num_linear_in = num_last_conv_channels \
        * in_width // pooling_factor \
        * in_height // pooling_factor
    self.encoder = nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, num_last_conv_channels, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(num_linear_in, out_size)
    )
  
  def forward(self, x):
    # change from (N, W, H, C) to (N, C, W, H)
    x = x.permute(0, 3, 1, 2)
    x = self.encoder(x)
    return x


class JointModel(nn.Module):
  def __init__(self, vocab_size, word_embedding_size, image_size, joint_embedding_size):
    super().__init__()
    self.text_enc = TextEncoder(vocab_size, word_embedding_size, joint_embedding_size)
    self.image_enc = ImageEncoder(image_size, image_size, 3, joint_embedding_size)
  
  def forward(self, x):
    x_text_positive, x_image_positive, x_text_anchor, x_image_anchor, x_text_negative, x_image_negative = x
    x_text_anchor = self.text_enc(x_text_anchor)
    x_text_positive = self.text_enc(x_text_positive)
    x_text_negative = self.text_enc(x_text_negative)
    x_image_anchor = self.image_enc(x_image_anchor)
    x_image_positive = self.image_enc(x_image_positive)
    x_image_negative = self.image_enc(x_image_negative)

    return x_text_positive, x_image_positive, x_text_anchor, x_image_anchor, x_text_negative, x_image_negative


class JointModel_classification(nn.Module): # Just to keep it in so we can also keep the singular network testing
  def __init__(self, vocab_size, word_embedding_size, image_size, joint_embedding_size):
    super().__init__()
    self.text_enc = TextEncoder(vocab_size, word_embedding_size, joint_embedding_size)
    self.image_enc = ImageEncoder(64, 64, 3, joint_embedding_size)
    # Add a FCN at the end of the encoder as a placeholder classifier.
    self.dummy_classifier = nn.Sequential(
        nn.Linear(joint_embedding_size, 4),
        nn.Softmax(dim=1)
    )
  
  def forward(self, x):
    x_text, x_image = x
    x_text = self.text_enc(x_text)
    x_image = self.image_enc(x_image)
    return self.dummy_classifier(x_image)
