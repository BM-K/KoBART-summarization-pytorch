import torch.nn as nn
from kobart import get_pytorch_kobart_model
from transformers import BartForConditionalGeneration


class KoBARTConditionalGeneration(nn.Module):
    def __init__(self, args):
        super(KoBARTConditionalGeneration, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(get_pytorch_kobart_model())
        self.model.to(args.device)

    def forward(self, inputs):
        outs = self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=inputs['decoder_attention_mask'],
                          labels=inputs['labels'], return_dict=True)
        loss = outs.loss

        return loss