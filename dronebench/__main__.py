import argparse
import json
import os
import platform
from pathlib import Path

from .config import BASE, bits_of, read_spec
from .records import file_hash, matrix, run_name, source_hash, split_hash, summarize, write_json


def recipe_args(p):
    p.add_argument('--data', default='VisDrone.yaml')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--init', help='trusted local stock YOLO11n checkpoint; default scratch')
    p.add_argument('--hparams', help='JSON object of optimizer, learning rate and augmentation overrides')
    p.add_argument('--kind', choices=['benchmark', 'smoke'], default='benchmark')
    p.add_argument('--project', default='runs/five')


def spec_args(p):
    which = p.add_mutually_exclusive_group()
    which.add_argument('--id', help='five bits: dm_efs esod querydet rfla ugs')
    which.add_argument('--config', help='experiment JSON')


def recipe(a):
    r = {k: getattr(a, k) for k in ('data','epochs','batch','imgsz','seed','device','workers','init','kind')}
    if getattr(a, 'hparams', None):
        opts = json.loads(a.hparams)
        allowed = set('optimizer lr0 lrf momentum weight_decay warmup_epochs warmup_momentum warmup_bias_lr nbs box cls dfl cls_pw cos_lr mosaic close_mosaic mixup cutmix scale translate degrees shear perspective hsv_h hsv_s hsv_v fliplr flipud'.split())
        if not isinstance(opts, dict) or set(opts) - allowed:
            raise ValueError('unknown train hparams; supported: ' + ', '.join(sorted(allowed)))
        r['hparams'] = json.dumps(opts, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return r


def check_device(device):
    if ',' in device or int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise ValueError('this reference pipeline supports one device per run, not DDP')
    return '' if device == 'auto' else device


def train(a):
    import torch
    import torchvision
    import ultralytics
    from .engine import ExperimentTrainer
    spec, r = read_spec(a.id, a.config), recipe(a)
    if a.imgsz < 64 or a.imgsz % 32 or a.epochs <= 0 or a.batch <= 0:
        raise ValueError('use resolution >=64 divisible by 32, positive epochs and explicit positive batch')
    init = str(Path(a.init).resolve(strict=True)) if a.init else False
    overrides = dict(model='yolo11n.yaml', task='detect', data=a.data, epochs=a.epochs,
                     pretrained=init, batch=a.batch, imgsz=a.imgsz, seed=a.seed,
                     device=check_device(a.device), workers=a.workers, project=str(Path(a.project).resolve()),
                     name=run_name(spec, r), amp=False, plots=False, optimizer='SGD', patience=0,
                     max_det=500, conf=0.001, iou=0.7)
    overrides.update(json.loads(r.get('hparams', '{}')))
    manifest = Path(a.data).parent / 'manifest.json'
    if a.kind == 'benchmark' and Path(a.data).is_file() and manifest.is_file():
        meta = json.loads(manifest.read_text())
        if isinstance(meta, dict) and str(meta.get('purpose', '')).startswith('engineering smoke only'):
            raise ValueError('the engineering fixture is not benchmark data; use --kind smoke')
    target = Path(overrides['project']) / overrides['name']
    target.mkdir(parents=True, exist_ok=False)
    overrides['exist_ok'] = True
    tr = ExperimentTrainer(spec, overrides)
    from ultralytics.utils.torch_utils import get_cpu_info
    hardware = torch.cuda.get_device_name(tr.device) if tr.device.type == 'cuda' else get_cpu_info()
    receipt = {'id': bits_of(spec), 'base_commit': BASE, 'source_sha256': source_hash(), 'config': spec,
               'recipe': r, 'status': 'started', 'data_yaml': str(tr.args.data),
               'data_sha256': file_hash(tr.args.data), 'init_sha256': file_hash(init) if init else None,
               'device': str(tr.device), 'hardware': hardware, 'python': platform.python_version(), 'torch': str(torch.__version__),
               'torchvision': str(torchvision.__version__), 'ultralytics': ultralytics.__version__}
    path = tr.save_dir / 'experiment.json'
    write_json(path, receipt)
    tr.train()
    complete = tr.epoch + 1 == a.epochs == tr.epoch_attempts and tr.batch_size == a.batch and tr.optimizer_steps > 0
    receipt.update(status='trained' if complete else 'noncomparable', completed_epochs=tr.epoch + 1,
                   epoch_attempts=tr.epoch_attempts, effective_batch=tr.batch_size, optimizer_steps=tr.optimizer_steps,
                   best=str(tr.best), last=str(tr.last), best_sha256=file_hash(tr.best) if tr.best.is_file() else None,
                   train_split_sha256=split_hash(tr.train_loader.dataset), val_split_sha256=split_hash(tr.test_loader.dataset))
    write_json(path, receipt)
    print(f'Run receipt: {path}\nValidate: python -m dronebench val --weights "{tr.best}" --data "{a.data}" --imgsz {a.imgsz} --batch {a.batch} --device {a.device}')


def validate(a):
    from .evaluation.validator import FixedBinValidator
    w = Path(a.weights).expanduser().resolve(strict=True)
    if w.suffix != '.pt':
        raise ValueError('validation requires a trusted local .pt checkpoint')
    folder = w.parent.parent if w.parent.name == 'weights' else w.parent
    validator = FixedBinValidator(args=dict(model=str(w), task='detect', mode='val', split=a.split, data=a.data,
                                           imgsz=a.imgsz, batch=a.batch, device=check_device(a.device), workers=0,
                                           max_det=500, conf=.001, iou=.7, rect=True, plots=False,
                                           project=str(folder / 'validation'), name=f'{a.split}-r{a.imgsz}'))
    validator(model=str(w))
    met = validator.metrics
    meta = {'weights': str(w), 'weights_sha256': file_hash(w), 'source_sha256': source_hash(), 'data': a.data,
            'data_sha256': file_hash(validator.data['yaml_file']), 'imgsz': a.imgsz, 'split': a.split, 'batch': a.batch,
            'device': str(validator.device), 'protocol': met.fixed_bin_protocol, 'split_sha256': split_hash(validator.dataloader.dataset),
            'mAP50-95': float(met.box.map), 'bins': met.fixed_bins, 'images': validator.seen}
    output = Path(a.output) if a.output else validator.save_dir / 'measurement.json'
    write_json(output, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2), '\nReport:', output)
    receipt = folder / 'experiment.json'
    if receipt.is_file() and w.name == 'best.pt':
        run = json.loads(receipt.read_text())
        r = run['recipe']
        if (run['status'] == 'trained' and run['completed_epochs'] == r['epochs'] and run.get('best_sha256') == meta['weights_sha256']
                and run['source_sha256'] == meta['source_sha256'] and r['imgsz'] == a.imgsz and a.split == 'val'
                and run['data_sha256'] == meta['data_sha256'] and run['val_split_sha256'] == meta['split_sha256']):
            write_json(folder / (r['kind'] + '-val.json'), meta)


def main():
    parser = argparse.ArgumentParser(description='YOLO11n five-method experiments on the pinned main snapshot')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train'); recipe_args(p); spec_args(p)
    p = sub.add_parser('val'); p.add_argument('--weights', required=True); p.add_argument('--data', default='VisDrone.yaml')
    p.add_argument('--imgsz', type=int, default=640); p.add_argument('--device', default='auto')
    p.add_argument('--batch', type=int, default=8); p.add_argument('--split', choices=['val','test'], default='val'); p.add_argument('--output')
    p = sub.add_parser('matrix'); recipe_args(p); p.add_argument('--output', default='experiment_records/matrix.csv')
    p = sub.add_parser('summarize'); p.add_argument('--runs', default='runs/five'); p.add_argument('--output', default='measured.csv')
    p = sub.add_parser('smoke'); spec_args(p); p.add_argument('--all', action='store_true'); p.add_argument('--imgsz', type=int, default=96)
    p.add_argument('--output'); p.add_argument('--seed', type=int, default=17)
    a = parser.parse_args()
    if a.command == 'train':
        train(a)
    elif a.command == 'val':
        validate(a)
    elif a.command == 'matrix':
        matrix(recipe(a), a.project, a.output)
        print(f'Wrote 32 experiments: {a.output}')
    elif a.command == 'smoke':
        from .smoke import smoke
        if a.all and (a.id or a.config):
            parser.error('use --all or one experiment, not both')
        smoke(a)
    else:
        summarize(a.runs, a.output)


if __name__ == '__main__':
    main()
