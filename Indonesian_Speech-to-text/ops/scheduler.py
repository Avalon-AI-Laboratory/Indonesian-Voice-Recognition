import torch

class Scheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.000001,
        warmup_epochs=100,
        decay_epochs=9900,
        steps_per_epoch=370,
        last_epoch=-1,
    ):
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        epoch = self._step_count // self.steps_per_epoch
        warmup_lr = self.init_lr + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        decay_lr = max(
            self.final_lr,
            self.lr_after_warmup - (epoch - self.warmup_epochs) * (self.lr_after_warmup - self.final_lr) / self.decay_epochs
        )
        return [min(warmup_lr, decay_lr) for _ in self.base_lrs]
