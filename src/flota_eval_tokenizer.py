from .flota import FlotaTokenizer
import numpy as np

class FlotaEvalTokenizer(FlotaTokenizer):
    
    def __init__(self, model, k, strict, mode):
        super().__init__(model, k, strict, mode)

    def remove_special_char(self, tokens):
        # removes the model specific special prefix characters
        return [token.lstrip(self.special) for token in tokens]

    def tokenize_no_special(self, word):
        # tokenizes the word and then removes the special characters from the tokens
        return self.remove_special_char(self.tokenize(word))

    def tokenize_no_special_batch(self, words):
        # tokenizes each word in the list and removes the special chararacters from the individual tokens
        return [self.tokenize_no_special(word) for word in words]

    def filter_ladec(self, dataset):
        # filters the dataset according to the model vocab
        # the two compounds have to be in the vocab while the normal entrie word is not
        # only needs to be used once for the model is not dependent on the mode or k
        word_not_in_vocab = np.array([word not in self.vocab and self.special + word not in self.vocab for word in dataset['stim']])
        comp1_in_vocab = np.array([word in self.vocab or self.special + word in self.vocab for word in dataset['c1']])
        comp2_in_vocab = np.array([word in self.vocab or self.special + word in self.vocab for word in dataset['c2']])
        return dataset[comp1_in_vocab & comp2_in_vocab & word_not_in_vocab]

    def tokenize_batch(self, words):
        return [self.tokenize(word) for word in words]