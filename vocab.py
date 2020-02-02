import re
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
    Special characters are removed and all words converted to lower-case.
    All unknown words are mapped to OOV_index and the sequence is terminated
    with the special marker END_index.
    """

    def map_word(w):
      if w in self.indices_by_word:
        return self.indices_by_word[w]
      elif w.endswith('s') and w[:-1] in self.indices_by_word:
        return self.indices_by_word[w[:-1]]
      else:
        return self.OOV_index

    input = re.sub(r"[^a-zA-Z\s]+", "", input.lower())
    input = re.sub(r"\s+", " ", input)
    input_words = input.split(" ")
    return [map_word(w) for w in input_words] \
           + [self.END_index]


  def seq_to_str(self, indices: Iterable[int]):
    """Converts a sequence of word indices back to a string, using placeholders for PAD, OOV and END."""

    return " ".join(self.words_by_index[i] for i in indices)
