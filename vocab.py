from typing import Iterable

PAD = "."
OOV = "[OOV]"
END = "[END]"

class Vocabulary:
  def __init__(self, words: Iterable[str]):
    # index 0 is the padding character
    self.words_by_index = [PAD] + sorted(words) + [OOV, END]
    self.indices_by_word = {word: i for i, word in enumerate(self.words_by_index)}
    self.size = len(self.words_by_index)
    self.OOV_index = self.indices_by_word[OOV]
    self.END_index = self.indices_by_word[END]


  def str_to_seq(self, input: str):
    """
    Converts a string to a sequence of integers representing its word indices.
    All unknown words will be mapped to OOV_index and the sequence will be terminated
    with the special marker END_index.
    """

    input_words = input.split(" ")
    return [self.indices_by_word.get(w, self.OOV_index) for w in input_words] \
           + [self.END_index]


  def seq_to_str(self, indices: Iterable[int]):
    """Converts a sequence of word indices back to a string, using placeholders for PAD, OOV and END."""

    return " ".join(self.words_by_index[i] for i in indices)
