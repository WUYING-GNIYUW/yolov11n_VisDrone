import json
from copy import copy

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

from .config import normalize
from .model import build_model


def _start_epoch(trainer):
    trainer.epoch_attempts += 1
    esod = getattr(unwrap_model(trainer.model), 'esod', None)
    if esod is not None:
        esod.epoch = trainer.epoch


def _finish_epoch(trainer):
    model = unwrap_model(trainer.model)
    if getattr(model, 'dm', None) is not None and trainer.ema is not None:
        dest = dict(trainer.ema.ema.dm.named_buffers())
        for name, buf in model.dm.named_buffers():
            if name.rsplit('.', 1)[-1] == 'votes':
                dest[name].copy_(buf)
    diag = getattr(model, 'last_diagnostics', {})
    if diag:
        row = {'epoch': trainer.epoch + 1, 'scope': 'last train batch', **diag}
        with (trainer.save_dir / 'routing.jsonl').open('a') as f:
            f.write(json.dumps(row, default=lambda x: x.detach().cpu().tolist() if isinstance(x, torch.Tensor) else str(x)) + '\n')


class ExperimentTrainer(DetectionTrainer):
    def __init__(self, spec, overrides):
        self.spec = normalize(spec)
        self.requested_batch = overrides['batch']
        self.optimizer_steps = 0
        self.epoch_attempts = 0
        super().__init__(overrides=overrides)
        self.add_callback('on_train_epoch_start', _start_epoch)
        self.add_callback('on_train_epoch_end', _finish_epoch)

    def _build_train_pipeline(self):
        if self.batch_size != self.requested_batch or getattr(self, '_oom_retries', 0):
            raise RuntimeError('batch changed on memory retry; rerun the whole cohort with one smaller batch')
        return super()._build_train_pipeline()

    def optimizer_step(self):
        super().optimizer_step()
        self.optimizer_steps += 1

    def get_model(self, cfg=None, weights=None, verbose=True):
        return build_model(self.spec, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose,
                           weights=weights, names=self.data['names'])

    def get_validator(self):
        from .evaluation.validator import FixedBinValidator
        return FixedBinValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
