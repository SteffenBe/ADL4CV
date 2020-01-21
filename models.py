import torch
from torch import nn
from datetime import datetime
import numpy as np
import pickle
import sys
if 'google.colab' not in sys.modules:
    from text_creation import description_templates, instructions, modifications_shape, modifications_color

shapes = ["square", "triangle", "star", "ellipse"]
colors = ["red", "green", "blue", "purple"]


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

    def __init__(self, vocab_words, word_embedding_size, lstm_size, out_size, path_to_glove="repo/glove.6B.50d.txt", train_word_embeddings=True):
        super().__init__()
        if path_to_glove:
            self.word_embedding = embedding_layer(vocab_words, embed_dim=word_embedding_size, path_to_glove=path_to_glove, trainable=train_word_embeddings)
        else:
            self.word_embedding = nn.Embedding(len(vocab_words), word_embedding_size, padding_idx=0)
            if not train_word_embeddings:
                print("WARNING: Not using pretrained embeddings, but train_word_embeddings is set to False! Will be limited by random fixed embeddings ...")
                self.word_embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(word_embedding_size, lstm_size, batch_first=True)
        self.final_linear = nn.Linear(lstm_size, out_size)

    def forward(self, x):
        x = self.word_embedding(x)
        x, (hidden, cell) = self.rnn(x)
        # Use last output of the sequence (final embedding).
        x = x[:, -1, :]
        x = self.final_linear(x)
        return x


class ImageEncoder(nn.Module):
    """The image encoder takes an input image and outputs the joint image embedding (J).
  
  Input shape: (N, in_resolution, in_resolution, in_channels)
  Output shape: (N, out_size)
  """

    def __init__(self, in_resolution, in_channels, conv_layers, out_size):
        super().__init__()
        
        self.cnn = nn.Sequential()

        pooling_factor = 1
        all_channels = [in_channels] + conv_layers
        all_channel_pairs = zip(all_channels[:-1], all_channels[1:])
        for i, (c_in, c_out) in enumerate(all_channel_pairs):
            self.cnn.add_module(name="conv{0}".format(i),
                    module=nn.Conv2d(c_in, c_out, 3, padding=1))
            self.cnn.add_module(name="act{0}".format(i),
                    module=nn.ReLU())
            # Add max pooling after every other CONV layer.
            #if i > 0 and i % 2 == 1:
            if True:
                pooling_factor *= 2
                self.cnn.add_module(name="pool{0}".format(i),
                        module=nn.MaxPool2d(2))
        
        num_linear_in = all_channels[-1] \
                        * in_resolution // pooling_factor \
                        * in_resolution // pooling_factor
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_linear_in, out_size)
        )

    def forward(self, x):
        # change from (N, W, H, C) to (N, C, W, H)
        x = x.permute(0, 3, 1, 2)
        x = self.cnn(x)
        x = self.linear(x)
        return x


class JointModel(nn.Module):
    def __init__(self, vocab_words, image_size, joint_embedding_size,
            word_embedding_size=51, path_to_glove="repo/glove.6B.50d.txt", train_word_embeddings=True,
            lstm_size=64,
            conv_layers=[32, 32, 32, 64, 64]):
        super().__init__()
        self.text_enc = TextEncoder(vocab_words, word_embedding_size, lstm_size, joint_embedding_size, path_to_glove, train_word_embeddings)
        self.image_enc = ImageEncoder(image_size, 3, conv_layers, joint_embedding_size)

    def forward(self, x):
        x_text_positive, x_image_positive, x_text_anchor, x_image_anchor, x_text_negative, x_image_negative = x
        x_text_anchor = self.text_enc(x_text_anchor)
        x_text_positive = self.text_enc(x_text_positive)
        x_text_negative = self.text_enc(x_text_negative)
        x_image_anchor = self.image_enc(x_image_anchor)
        x_image_positive = self.image_enc(x_image_positive)
        x_image_negative = self.image_enc(x_image_negative)

        return x_text_positive, x_image_positive, x_text_anchor, x_image_anchor, x_text_negative, x_image_negative


def make_weights_matrix(vocabulary=[], path_to_glove="glove.6B.50d.txt", embed_dim=51):
    # embed_dim = 51
    if path_to_glove.split(".")[-1] == "txt":
        glove_dict = {}
        with open(path_to_glove, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                if word not in vocabulary:
                    continue
                vector = np.array(line[1:]).astype(np.float)
                glove_dict[word] = vector

    elif path_to_glove.split(".")[-1] in ["pkl", "pickle", "pck", "pcl"]:
        glove_dict = load_obj(path_to_glove)

    else:
        raise ValueError("File %s could not be loaded as glove data" % path_to_glove)

    matrix_len = len(vocabulary)
    weights_matrix = np.zeros((matrix_len, embed_dim))
    not_found_words = []
    found_words = vocabulary.copy()
    for i, word in enumerate(vocabulary):
        if i not in [0, matrix_len - 1, matrix_len - 2]:
            try:
                weights_matrix[i, 0:embed_dim - 1] = glove_dict[word]
            except KeyError:
                not_found_words.append(word)
                found_words.remove(word)
                weights_matrix[i, 0:embed_dim - 1] = np.random.normal(scale=0.6, size=(embed_dim - 1,))

    weights_matrix[-2, -1] = 0.5
    weights_matrix[-1, -1] = 1

    print("The following words were found in glove data: %s" % found_words)
    print("The following words were not found in glove data: %s" % not_found_words)

    return weights_matrix


def create_emb_layer(weights_matrix, trainable=False):

    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix)})
    if not trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings


def embedding_layer(vocab_word_list, path_to_glove="glove.6B.50d.txt", trainable=False, embed_dim=51):

    weights_matrix = make_weights_matrix(vocab_word_list, path_to_glove, embed_dim=embed_dim)

    emb_layer, num_embeddings = create_emb_layer(weights_matrix, trainable=trainable)

    return emb_layer


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


def make_relevant_glove_dict(vocabulary_words=[], path_to_glove="glove.6B.50d.txt", take_all=False):

    if not take_all and vocabulary_words == []:
        all_strings = description_templates + instructions + modifications_shape + modifications_color + shapes + colors

        vocabulary_words = []
        for text in all_strings:
            words = text.split(" ")
            for word in words:
                if word[0] == "{" and word[-1] == "}":
                    continue
                if word not in vocabulary_words:
                    vocabulary_words.append(word)
        print("%s unique used words found." % len(vocabulary_words))

    glove_dict = {}
    with open(path_to_glove, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if not take_all and word not in vocabulary_words:
                continue
            vector = np.array(line[1:]).astype(np.float)
            glove_dict[word] = vector

    return glove_dict


def save_relevant_glove_dict(vocabulary_words=[], save_glove_dict_as="relevant_glove_dict.pkl",
                             path_to_glove="glove.6B.50d.txt"):

    glove_dict = make_relevant_glove_dict(vocabulary_words=vocabulary_words, path_to_glove=path_to_glove)

    print("Saving glove dictionary with %s entries as '%s.pkl'" % (len(glove_dict), save_glove_dict_as))
    save_obj(glove_dict, save_glove_dict_as)


def save_all_glove_dict(save_glove_dict_as="all_glove_dict.pkl", path_to_glove="glove.6B.50d.txt"):

    glove_dict = make_relevant_glove_dict(path_to_glove=path_to_glove, take_all=True)

    print("Saving glove dictionary with %s entries as '%s'" % (len(glove_dict), save_glove_dict_as))
    save_obj(glove_dict, save_glove_dict_as)


def save_obj(obj, name):
    if name.split(".")[-1] not in ["pkl", "pickle", "pck", "pcl"]:
        name = name + '.pkl'
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    if name.split(".")[-1] not in ["pkl", "pickle", "pck", "pcl"]:
        name = name + '.pkl'
    with open(name, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    save_relevant_glove_dict()
    # test_vocab = [".", "a", "all", "and", "any"]
    # a = make_weights_matrix(vocabulary=test_vocab, path_to_glove="relevant_glove_dict.pkl")
    # print(a)
    # check_glove(word_string="draft")
    # example_vocabulary = [".", "test", "asdfasdfa2 fgb", "asdfasdireuireuirue", "OOV", "END"]
    # print(make_weights_matrix(vocabulary=example_vocabulary))
