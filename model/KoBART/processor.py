import logging
from apex import amp
import torch.nn as nn
from tqdm import tqdm
import torch.quantization
import torch.optim as optim
from model.utils import Metric
from data.dataloader import get_loader
from model.KoBART.model import KoBARTConditionalGeneration
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class Processor():

    def __init__(self, args):
        self.args = args
        self.config = None
        self.metric = Metric(args)
        self.model_checker = {'early_stop': False,
                              'early_stop_patient': 0,
                              'best_valid_loss': float('inf')}
        self.model_progress = {'loss': -1, 'iter': -1, 'acc': -1}
        self.sorted_path = args.path_to_save + args.ckpt

    def run(self, inputs, mode=None):
        loss = self.config['model'](inputs, mode)

        return loss

    def progress(self, loss):
        self.model_progress['loss'] += loss
        self.model_progress['iter'] += 1

    def return_value(self):
        loss = self.model_progress['loss'].data.cpu().numpy() / self.model_progress['iter']

        return loss

    def get_object(self, tokenizer, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.args.lr)

        return criterion, optimizer

    def get_scheduler(self, optim, train_loader):
        train_total = len(train_loader) * self.args.epochs
        scheduler = get_linear_schedule_with_warmup(
                    optim,
                    num_warmup_steps=self.args.warmup_ratio*train_total,
                    num_training_steps=train_total)

        return scheduler

    def model_setting(self):
        loader, tokenizer = get_loader(self.args, self.metric)

        model = KoBARTConditionalGeneration(self.args, tokenizer)
        model.to(self.args.device)

        criterion, optimizer = self.get_object(tokenizer, model)
        
        if self.args.test == 'False':
            scheduler = self.get_scheduler(optimizer, loader['train'])
        else:
            scheduler = None

        config = {'loader': loader,
                  'optimizer': optimizer,
                  'criterion': criterion,
                  'scheduler': scheduler,
                  'tokenizer': tokenizer,
                  'args': self.args,
                  'model': model}

        if config['args'].fp16 == 'True' and config['args'].test == 'False':
            config['model'], config['optimizer'] = amp.initialize(
                config['model'], config['optimizer'], opt_level=config['args'].opt_level)

        self.config = config

        return self.config

    def train(self):
        self.config['model'].train()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        for step, batch in enumerate(tqdm(self.config['loader']['train'])):
            self.config['optimizer'].zero_grad()

            inputs = batch
            loss = self.run(inputs, mode='train')

            if self.args.fp16 == 'True':
                with amp.scale_loss(loss, self.config['optimizer']) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.config['optimizer'].step()
            self.config['scheduler'].step()
            self.progress(loss.data)

        return self.return_value()

    def valid(self):
        self.config['model'].eval()
        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(self.config['loader']['valid']):

                inputs = batch
                loss = self.run(inputs, mode='valid')

                self.progress(loss.data)

        return self.return_value()

    def test(self):
        self.config['model'].load_state_dict(torch.load(self.sorted_path))
        self.config['model'].eval()

        self.model_progress = self.model_progress.fromkeys(self.model_progress, 0)

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.config['loader']['test'])):

                inputs = batch
                self.metric.generation(self.config, inputs)

        return self.metric.avg_rouge()
