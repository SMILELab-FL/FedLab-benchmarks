
import sys
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from torch.utils.data import Dataset
from leaf.nlp_utils.tokenizer import Tokenizer


class Sent140Dataset(Dataset):

    def __init__(self, client_id: int, client_str: str, data: list, targets: list,
                 is_to_tokens=True, tokenizer=None):
        """get `Dataset` for sent140 dataset

        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
            is_to_tokens (bool, optional), if tokenize data by using tokenizer
            tokenizer (Tokenizer, optional), tokenizer
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self.data_token = []
        self.data_seq = []
        self.targets_tensor = []
        self.vocab = None
        self.tokenizer = tokenizer if tokenizer else Tokenizer('normal')
        self.maxlen = None

        self._process_data_target()
        if is_to_tokens:
            self._data2token()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = [e[4] for e in self.data]
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def _data2token(self):
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def token2seq(self, vocab: 'Vocab', maxlen: int):
        """transform token data to indices sequence by `vocab`

        Args:
            vocab (fedlab_benchmark.leaf.nlp_utils.vocab): vocab for data_token
            maxlen (int): max length of sentence

        Returns:
            list of integer list for data_token, and a list of tensor target
        """
        if len(self.data_seq) > 0:
            self.data_seq.clear()
            self.targets_tensor.clear()
        self.vocab = vocab
        self.maxlen = maxlen
        assert self.data_token is not None
        for tokens in self.data_token:
            self.data_seq.append(self.__encode_tokens(tokens))
        for target in self.targets:
            self.targets_tensor.append(torch.tensor(target))

    def __encode_tokens(self, tokens) -> torch.Tensor:
        """encode `maxlen` length for token_data to get indices list in `self.vocab`
        if one sentence length is shorter than maxlen, it will use pad word for padding to maxlen
        if one sentence length is longer than maxlen, it will cut the first max_words words

        Args:
            tokens (list[str]): data after tokenizer

        Returns:
            integer list of indices with `maxlen` length for tokens input
        """
        pad_word = 0
        x = [pad_word for _ in range(self.maxlen)]
        temp = tokens[:self.maxlen]
        for idx, word in enumerate(temp):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, item):
        return self.data_seq[item], self.targets_tensor[item]
