"""Word symbols class"""

import os
import torch
import torch.distributed as dist
import torchtext

from bootleg.symbols.constants import PAD, PAD_ID, UNK_ID
import bootleg.utils.utils as utils

class WordSymbols:
    """
    WordSymbols class that tokenizes sentences.
    """
    def __init__(self, args, is_writer, distributed):
        self.vocab = {}
        self.vocab_rev = {}
        self.word_embedding_dim = -1
        self.num_words = -1
        self.unk_id = UNK_ID
        self.pad_id = PAD_ID
        self.is_bert = False

    def tokenize(self, text):
        return text.split(" ")

    def convert_tokens_to_ids(self, tokens):
        processed_sent = []
        # we add 1 to indices to support UNK_ID in first row
        for w in tokens:
            if w == PAD:
                idx = self.pad_id
            # try original casing of word first
            elif w in self.vocab:
                idx = self.vocab[w]+1
            # revert to lower case version if word not found
            elif w.lower() in self.vocab:
                idx = self.vocab[w.lower()]+1
            # if still not found return unk
            else:
                idx = self.unk_id
            processed_sent.append(idx)
        return processed_sent

    def convert_ids_to_tokens(self, indices):
        #support for batched or non-batched indices
        if len(indices.shape) == 1:
            indices = indices.unsqueeze(0)
        return [[self.vocab_rev[int(i-1)] if (i != self.pad_id and i != self.unk_id) else '_' for i in idxs] for idxs in indices]


class BERTWordSymbols(WordSymbols):
    """BERT tokenizer class"""
    def __init__(self, args, is_writer, distributed):
        super(BERTWordSymbols, self).__init__(args, is_writer, distributed)
        self.is_bert = True
        self.unk_id = None
        self.pad_id = 0
        cache_dir = args.word_embedding.cache_dir
        utils.ensure_dir(cache_dir)
        # import torch
        # import os
        # from transformers import BertTokenizer
        # cache_dir = "./"
        # tokenizer = BertTokenizer.from_pretrained('bert-base-cased', cache_dir=cache_dir, do_lower_case=False)
        # torch.save(tokenizer, os.path.join(cache_dir, "bert_base_cased_tokenizer.pt"))
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, do_lower_case=True)
        # torch.save(tokenizer, os.path.join(cache_dir, "bert_base_uncased_tokenizer.pt"))
        if args.word_embedding.use_lower_case:
            self.tokenizer = torch.load(os.path.join(cache_dir, "bert_base_uncased_tokenizer.pt"))
        else:
            self.tokenizer = torch.load(os.path.join(cache_dir, "bert_base_cased_tokenizer.pt"))
        self.vocab = self.tokenizer.vocab
        self.num_words = len(self.vocab)
        self.word_embedding_dim = 768

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, indices):
        # support for batched or non-batched indices
        if len(indices.shape) == 1:
            indices = indices.unsqueeze(0)
        return [self.tokenizer.convert_ids_to_tokens(idxs.to('cpu').numpy()) for idxs in indices]


class GLOVEWordSymbols(WordSymbols):
    """Glove word tokenizers"""
    def __init__(self, args, is_writer, distributed):
        super(GLOVEWordSymbols, self).__init__(args, is_writer, distributed)
        self.is_bert = False
        if not os.path.exists(os.path.join(args.emb_dir, 'glove.840B.300d.txt.pt')):
            raise ValueError(
                'Word embedding is not in cache directory. In python shell, run \nimport torchtext\nword_embeddings = torchtext.vocab.GloVe(cache=\'embs\')')
        word_embeddings = torchtext.vocab.GloVe(cache=args.emb_dir)
        self.vocab = word_embeddings.stoi
        self.vocab_rev = word_embeddings.itos
        self.word_embedding_dim = word_embeddings.vectors.shape[1]
        self.num_words = word_embeddings.vectors.shape[0]
        del word_embeddings


class CustomWordSymbols(WordSymbols):
    """Custom word tokenizers that uses torchtext"""
    def __init__(self, args, is_writer, distributed):
        super(CustomWordSymbols, self).__init__(args, is_writer, distributed)
        self.is_bert = False
        if args.overwrite_preprocessed_data:
            # TODO: this still isn't thread safe for prep
            path_pt = os.path.join(args.emb_dir, args.word_embedding.custom_vocab_embedding_file) + ".pt"
            if utils.exists_dir(path_pt) and is_writer:
                os.remove(path_pt)
            if distributed:
                dist.barrier()
        word_embeddings = torchtext.vocab.Vectors(
            args.word_embedding.custom_vocab_embedding_file, cache=args.emb_dir)
        self.vocab = word_embeddings.stoi
        self.word_embedding_dim = word_embeddings.vectors.shape[1]
        self.num_words = word_embeddings.vectors.shape[0]
        del word_embeddings
