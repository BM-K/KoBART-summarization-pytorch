import os
import torch
import logging
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.args = args

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
            cur_kv = key + ': ' + value + '\n'
            config += cur_kv
        config += 'Epoch: ' + cp["ep"] + '\t' + 'Valid loss: ' + cp['vl'] + '\n'

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

        self.draw_graph(cp)
        self.performance_check(cp)