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

    def __init__(self, vocab_words, word_embedding_size, out_size, path_to_glove="repo/glove.6B.50d.txt"):
        super().__init__()
        self.word_embedding, embedding_size = embedding_layer(vocab_words, embed_dim=word_embedding_size, path_to_glove=path_to_glove)
        self.rnn = nn.LSTM(embedding_size, out_size, batch_first=True)
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
    def __init__(self, vocab_words, image_size, joint_embedding_size,  word_embedding_size=51, path_to_glove="repo/glove.6B.50d.txt"):
        super().__init__()
        self.text_enc = TextEncoder(vocab_words, word_embedding_size, joint_embedding_size, path_to_glove=path_to_glove)
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


def make_weights_matrix(vocabulary=None, path_to_glove="glove.6B.50d.txt", embed_dim=51):
    #embed_dim = 51
    glove_dict = {}
    skip = True
    with open(path_to_glove, 'rb') as f:
        for l in f:
            # if skip == True:
            #     skip = False
            #     continue
            print(".............")
            print(l)
            line = l.decode().split()
            print("XXXXXXXXXXXXXXXXXXX")
            print(line)
            # print(line)
            # word = line[0]
            # if word not in vocabulary:
            #     continue
            # vector = np.array(line[1:]).astype(np.float)
            # glove_dict[word] = vector

    print([key for key, value in glove_dict.items()])

    matrix_len = len(vocabulary)
    weights_matrix = np.zeros((matrix_len, embed_dim))
    not_found_words = []
    # found_words = vocabulary
    for i, word in enumerate(vocabulary):
        if i not in [0, matrix_len - 1, matrix_len - 2]:
            try:
                weights_matrix[i, 0:embed_dim - 1] = glove_dict[word]
            except KeyError:
                not_found_words.append(word)
                # found_words.remove(word)
                weights_matrix[i, 0:embed_dim - 1] = np.random.normal(scale=0.6, size=(embed_dim - 1,))

    weights_matrix[-2, -1] = 0.5
    weights_matrix[-1, -1] = 1

    # print("The following words were found in glove data: %s" % found_words)
    print("The following words were not found in glove data: %s" % not_found_words)


    return weights_matrix

def create_emb_layer(weights_matrix, trainable=False):

    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

def embedding_layer(vocab_word_list, path_to_glove="glove.6B.50d.txt", trainable=False, embed_dim=51):

    weights_matrix = make_weights_matrix(vocab_word_list, path_to_glove, embed_dim=embed_dim)

    emb_layer, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, trainable=trainable)

    return emb_layer, embedding_dim


def check_glove(word_string, path_to_glove="glove.6B.50d.txt", glove_list=None):

    if glove_list is None:
        glove_list = []
        with open(path_to_glove, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                glove_list.append(word)

    if word_string in glove_list:
        print("Word: %s found in raw glove" % word_string)

    glove_list_lower = [word.lower() for word in glove_list]

    if word_string.lower() in glove_list_lower and word_string != word_string.lower():
        print("Word: %s converted to %s and found in lower case glove list" % (word_string, word_string.lower()))



if __name__ == "__main__":
    test_vocab = [".", "a", "all", "and", "any"]
    make_weights_matrix(vocabulary=test_vocab)
    # check_glove(word_string="draft")
    # example_vocabulary = [".", "test", "asdfasdfa2 fgb", "asdfasdireuireuirue", "OOV", "END"]
    # print(make_weights_matrix(vocabulary=example_vocabulary))
