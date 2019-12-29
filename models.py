import torch
from torch import nn
from datetime import datetime
import numpy as np


def count_params(model):
    """Returns the total number of trainable parameters in a torch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(label, model, verbose=True):
    f = '{0}--{1}--{2}.pt'.format(label, count_params(model), datetime.utcnow().strftime('%Y-%m-%dT%H-%M'))
    torch.save(model.state_dict(), f)
    if verbose:
        print('Model saved at "{}"'.format(f))


class TextEncoder(nn.Module):
    """The text encoder takes a sequence of word indices and outputs
  the joint sequence embedding (J)."""

    def __init__(self, vocab_size, word_embedding_size, out_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(word_embedding_size, out_size, batch_first=True)
        self.final_linear = nn.Linear(out_size, out_size)

    def forward(self, x):
        x = self.word_embedding(x)
        x, (hidden, cell) = self.rnn(x)
        # Use last output of the sequence (final embedding).
        x = x[:, -1, :]
        x = self.final_linear(x)
        return x


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


class JointModel_classification(nn.Module):  # Just to keep it in so we can also keep the singular network testing
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


def make_weights_matrix(vocabulary=None, path_to_glove="glove.6B.50d.txt"):
    embed_dim = 51
    glove_dict = {}
    with open(path_to_glove, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if word not in vocabulary:
                continue
            vector = np.array(line[1:]).astype(np.float)
            glove_dict[word] = vector

    matrix_len = len(vocabulary)
    weights_matrix = np.zeros((matrix_len, embed_dim))
    not_found_words = []
    for i, word in enumerate(vocabulary):
        if i not in [0, matrix_len - 1, matrix_len - 2]:
            try:
                weights_matrix[i, 0:embed_dim - 1] = glove_dict[word]
            except KeyError:
                not_found_words.append(word)
                weights_matrix[i, 0:embed_dim - 1] = np.random.normal(scale=0.6, size=(embed_dim - 1,))

    weights_matrix[-2, -1] = 0.5
    weights_matrix[-1, -1] = 1

    print("The following words were not found in glove data: %s" % not_found_words)

    return weights_matrix


if __name__ == "__main__":
    example_vocabulary = [".", "test", "asdfasdfa2 fgb", "asdfasdireuireuirue", "OOV", "END"]
    print(make_weights_matrix(vocabulary=example_vocabulary))
