"""A simple tokenizer for text instructions.

Word-level and whitespace-split: good enough for the one-sentence commands this project
uses, and small enough to read in one sitting.
"""


class SimpleTokenizer:
    """Maps words to ids. The vocabulary is built from the collected instructions."""

    def __init__(self, vocab=None):
        # vocab: dict token -> id
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        if vocab is None:
            self.vocab = {self.pad_token: 0, self.unk_token: 1}
        else:
            self.vocab = vocab

    def build_from_texts(self, texts):
        """Add every unseen word to the vocabulary, in order of first appearance."""
        for txt in texts:
            for tok in txt.lower().split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

    def encode(self, text):
        """Text -> list of ids; unknown words get the <unk> id.

        This does not pad or truncate. Every sentence comes back at its own length, and
        padding them to a common width is the caller's job (see scripts/collect_data.py).
        """
        toks = text.lower().split()
        ids = [self.vocab.get(t, self.vocab[self.unk_token]) for t in toks]
        return ids
