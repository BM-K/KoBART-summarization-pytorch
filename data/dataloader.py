import re
import csv
import torch
import logging
import numpy as np
from kobart import get_kobart_tokenizer
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ModelDataLoader(Dataset):
    def __init__(self, file_path, args, metric, tokenizer):
        self.args = args
        self.metric = metric

        self.tokenizer = tokenizer
        self.file_path = file_path

        self.input_ids = []
        self.attention_mask = []
        self.decoder_input_ids = []
        self.decoder_attention_mask = []
        self.labels = []

        """
        init token, idx = <s>, 0
        pad token, idx = <pad>, 3
        unk token, idx = <unk>, 5
        eos token, idx = </s>, 1
        """

        self.init_token = self.tokenizer.bos_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.eos_token = self.tokenizer.eos_token

        self.init_token_idx = self.tokenizer.convert_tokens_to_ids(self.init_token)
        self.pad_token_idx = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        self.unk_token_idx = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.eos_token_idx = self.tokenizer.convert_tokens_to_ids(self.eos_token)
        self.ignore_index = -100

    def load_data(self, type):

        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = csv.reader(file, delimiter="\t", quotechar='"')

            for line in lines:
                check = self.data2tensor(line)

        assert len(self.input_ids) == \
               len(self.attention_mask) == \
               len(self.decoder_input_ids) ==\
               len(self.decoder_attention_mask) ==\
               len(self.labels)

    def preprocess_text(self, review):

        # Remove breaks
        review = re.sub("<br />", " ", review)

        # Remove non-letters and non-numbers
        alphanum = re.sub("[^A-Za-z0-9가-힣]", " ", review)

        # Convert to lowercase and split into individual words
        tokens = alphanum.lower().split()

        return (" ".join(tokens))

    def data2tensor(self, line):
        try:
            source, target =  line[0].strip(), line[1].strip()
        except IndexError:
            logger.info("Index Error")
            exit()
            return False

        input_ids = self.add_padding_data(self.tokenizer.encode(source))
        label_ids = self.tokenizer.encode(target) + [self.eos_token_idx]
        dec_input_ids = self.add_padding_data([self.pad_token_idx] + label_ids[:-1])
        label_ids = self.add_ignored_data(label_ids)

        input_ids = torch.tensor(input_ids)
        dec_input_ids = torch.tensor(dec_input_ids)
        label_ids = torch.tensor(label_ids)

        attention_mask = input_ids.ne(self.pad_token_idx).float()
        decoder_attention_mask = dec_input_ids.ne(self.pad_token_idx).float()

        self.input_ids.append(input_ids)
        self.attention_mask.append(attention_mask)
        self.decoder_input_ids.append(dec_input_ids)
        self.decoder_attention_mask.append(decoder_attention_mask)
        self.labels.append(label_ids)

        return True

    def add_padding_data(self, inputs):
        if len(inputs) < self.args.max_len:
            pad = np.array([self.pad_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.args.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.args.max_len:
            pad = np.array([self.ignore_index] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.args.max_len]

        return inputs

    def __getitem__(self, index):

        input_data = {'input_ids': self.input_ids[index].to(self.args.device),
                      'attention_mask': self.attention_mask[index].to(self.args.device),
                      'decoder_input_ids': self.decoder_input_ids[index].to(self.args.device),
                      'decoder_attention_mask': self.decoder_attention_mask[index].to(self.args.device),
                      'labels': self.labels[index].to(self.args.device)}
        
        return input_data

    def __len__(self):
        return len(self.labels)


def get_loader(args, metric):
    path_to_train_data = args.path_to_data + '/' + args.train_data
    path_to_valid_data = args.path_to_data + '/' + args.valid_data
    path_to_test_data = args.path_to_data + '/' + args.test_data

    tokenizer = get_kobart_tokenizer()

    if args.train == 'True' and args.test == 'False':
        train_iter = ModelDataLoader(path_to_train_data, args, metric, tokenizer)
        valid_iter = ModelDataLoader(path_to_valid_data, args, metric, tokenizer)
        train_iter.load_data('train')
        valid_iter.load_data('valid')

        loader = {'train': DataLoader(dataset=train_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True),
                  'valid': DataLoader(dataset=valid_iter,
                                      batch_size=args.batch_size,
                                      shuffle=True)}

    elif args.train == 'False' and args.test == 'True':
        test_iter = ModelDataLoader(path_to_test_data, args, metric, tokenizer)
        test_iter.load_data('test')

        loader = {'test': DataLoader(dataset=test_iter,
                                     batch_size=args.batch_size,
                                     shuffle=False)}

    else:
        logger.info("Error: None type loader")
        exit()

    return loader, tokenizer

if __name__ == '__main__':
    get_loader('test')
