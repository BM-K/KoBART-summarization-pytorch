import re
import os
import csv
import torch
import logging
from rouge import Rouge
from konlpy.tag import Mecab
from tensorboardX import SummaryWriter

mecab = Mecab()
logger = logging.getLogger(__name__)
writer = SummaryWriter()
REMOVE_CHAR_PATTERN = re.compile("[^A-Za-z0-9가-힣]")


class Metric():

    def __init__(self, args):
        self.args = args
        self.step = 0
        self.rouge = Rouge()
        self.rouge_scores = {'rouge-1': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-2': {'r': 0, 'p': 0, 'f': 0},
                             'rouge-l': {'r': 0, 'p': 0, 'f': 0}}

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def draw_graph(self, cp):
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])

    def performance_check(self, cp):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Valid Loss: {cp["vl"]:.4f}==')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def save_config(self, cp):
        config = "Config>>\n"
        for idx, (key, value) in enumerate(self.args.__dict__.items()):
            cur_kv = str(key) + ': ' + str(value) + '\n'
            config += cur_kv
        config += 'Epoch: ' + str(cp["ep"]) + '\t' + 'Valid loss: ' + str(cp['vl']) + '\n'

        with open(self.args.path_to_save+'config.txt', "w") as f:
            f.write(config)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + config['args'].ckpt

        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']
            torch.save(config['model'].state_dict(), sorted_path)
            self.save_config(cp)
            print(f'\n\t## SAVE valid_loss: {cp["vl"]:.4f} ##')
        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True
                writer.close()

        # self.draw_graph(cp)
        self.performance_check(cp)

    def result_file(self, config, source, ref, hyp):
        sorted_path = config['args'].path_to_save + 'result.tsv'
        with open(sorted_path, 'a', encoding='utf-8') as f:
            tw = csv.writer(f, delimiter='\t')
            tw.writerow([source, ref, hyp])

    def rouge_score(self, config, ref, hyp):
        # ref = ' '.join(config['tokenizer'].tokenize(ref.strip()))
        # hyp = ' '.join(config['tokenizer'].tokenize(hyp.strip()))
        ref = ' '.join(mecab.morphs(REMOVE_CHAR_PATTERN.sub(" ", ref.lower()).strip()))
        hyp = ' '.join(mecab.morphs(REMOVE_CHAR_PATTERN.sub(" ", hyp.lower()).strip()))

        """
        ref = ' '.join(config['tokenizer'].tokenize(
            REMOVE_CHAR_PATTERN.sub(" ", ref.lower()).strip()))
        ref = REMOVE_CHAR_PATTERN.sub(" ", ref.lower())

        hyp = ' '.join(config['tokenizer'].tokenize(
            REMOVE_CHAR_PATTERN.sub(" ", hyp.lower()).strip()))
        hyp = REMOVE_CHAR_PATTERN.sub(" ", hyp.lower())
        """
        score = self.rouge.get_scores(hyp, ref)[0]
        
        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] += score[metric][key]

        self.step += 1

    def avg_rouge(self):
        for metric, scores in self.rouge_scores.items():
            for key, value in scores.items():
                self.rouge_scores[metric][key] /= self.step

        return self.rouge_scores

    def generation(self, config, inputs):
        outputs = config['model'].model.generate(inputs['input_ids'],
                                                 max_length=self.args.max_len,
                                                 num_beams=5)

        for step, beam in enumerate(outputs):
            ref = config['tokenizer'].decode(inputs['decoder_input_ids'][step], skip_special_tokens=True)
            hyp = config['tokenizer'].decode(beam, skip_special_tokens=True)

            self.rouge_score(config, ref, hyp)