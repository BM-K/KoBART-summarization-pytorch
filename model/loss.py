import torch
import torch.nn as nn
import torch.nn.functional as F


class LossF():

    def __init__(self, args):
        self.args = args

    def base(self, config, logits, labels):

        return config['criterion'](logits, labels)

    def distilled(self, config, inputs, logits_s, labels):

        student_loss = config['criterion'](logits_s, labels)
        teacher_outputs = config['teacher'](inputs)

        loss_KD = F.kl_div(F.log_softmax(logits_s / config['args'].temperature, dim=1),
                           F.softmax(teacher_outputs / config['args'].temperature, dim=1), reduction="batchmean")

        loss = (1 - self.args.alpha) * student_loss + self.args.alpha * (
                self.args.temperature ** 2) * loss_KD

        return loss