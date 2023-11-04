import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss

class SmoothCTC_CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, model_output, dec_target):
        device = model_output.device
        mask = dec_target != 0
        T, N, C = model_output.shape[1], model_output.shape[0], model_output.shape[2]

        logits = F.log_softmax(model_output, dim=2).transpose(0, 1) # Expects (input_length, batch_sz, classes) dimension
        ctc_function = CTCLoss()
        input_lengths = torch.full(size=(N, ), fill_value = T, dtype = torch.long).to(device)
        target_lengths = torch.sum(mask, dim = 1).to(device)
        ctc_loss = ctc_function(logits, dec_target, input_lengths, target_lengths)

        true_dist = torch.zeros_like(logits.transpose(0, 1)).to(device)
        true_dist.fill_(self.smoothing / (C - 1))
        true_dist.scatter_(2, dec_target.unsqueeze(2).to(device).long(), 1 - self.smoothing)
        ce_smoothing_loss = (-true_dist * logits.transpose(0, 1)).sum(dim = 2)
        ce_smoothing_loss = (ce_smoothing_loss * mask.float().to(device)).sum() / mask.float().sum().to(device)

        loss = ctc_loss + ce_smoothing_loss
        return loss
