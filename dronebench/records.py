import csv
import hashlib
import json
import math
import shlex
from pathlib import Path

from .config import BASE, METHODS, bits_of, combinations, digest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_hash(dataset):
    import os
    import numpy as np
    files = [Path(f) for f in dataset.im_files]
    if not files or len(files) != len(dataset.labels):
        raise ValueError('expected nonempty YOLO split with one label record per image')
    root = Path(os.path.commonpath([str(f.parent) for f in files]))
    h = hashlib.sha256()
    for f, label in sorted(zip(files, dataset.labels), key=lambda pair: str(pair[0])):
        h.update(str(f.relative_to(root)).encode() + b'\0' + file_hash(f).encode())
        for key in ('cls', 'bboxes'):
            a = np.asarray(label[key], dtype='<f4')
            h.update(str(a.shape).encode() + b'\0' + a.tobytes())
        h.update(str((label.get('normalized'), label.get('bbox_format'))).encode())
    return h.hexdigest()


def source_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        rel = path.relative_to(root)
        if (any(p.startswith('.') or p in ('tests', 'checks') for p in rel.parts) or path.name.startswith('test_')
                or path.name in ('smoke.py', 'e2e_smoke.py', 'example.py')):
            continue
        h.update(str(rel).encode() + b'\0' + path.read_bytes())
    return h.hexdigest()


def run_name(spec, recipe):
    return f"{recipe['kind']}-{bits_of(spec)}-r{recipe['imgsz']}-s{recipe['seed']}-e{recipe['epochs']}-{digest([spec, recipe])}"


def write_json(path, data):
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(a) for k, a in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(a) for a in v]
        return None if isinstance(v, float) and not math.isfinite(v) else v
    Path(path).write_text(json.dumps(clean(data), indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=dict.fromkeys(k for row in rows for k in row))
        writer.writeheader()
        writer.writerows(rows)


def matrix(recipe, project, output):
    rows = []
    impl = source_hash()
    for spec in combinations():
        bits = bits_of(spec)
        name = run_name(spec, recipe)
        weight = str(Path(project) / name / 'weights' / 'best.pt')
        train = ['python', '-m', 'dronebench', 'train', '--id', bits, '--project', str(project)]
        for k, v in recipe.items():
            if v is not None:
                train.extend(['--' + k.replace('_','-'), str(v)])
        val = ['python', '-m', 'dronebench', 'val', '--weights', weight, '--data', recipe['data'],
               '--imgsz', str(recipe['imgsz']), '--batch', str(recipe['batch']), '--device', recipe['device']]
        rows.append({'id': bits, **{k: int(spec[k]) for k in METHODS}, 'order': spec['order'], 'base_commit': BASE,
                     'experiment_source_sha256': impl, 'config': json.dumps(spec, separators=(',', ':')),
                     'run': name, **recipe, 'protocol': 'YOLO mAP; NMS conf=.001 IoU=.7 cap=500; R50@640 conf=.25 cap=300',
                     'full_training': 'pending', 'mAP50-95': '', 'R_lt8': '', 'R_8to16': '', 'R_ge16': '',
                     'end_to_end_b1_p50_ms': '', 'end_to_end_b1_p95_ms': '', 'delta_mAP': '', 'interaction_mAP': '',
                     'train_command': shlex.join(train), 'val_command': shlex.join(val)})
    write_csv(output, rows)


def summarize(root, output):
    rows, groups = [], {}
    for path in sorted(Path(root).rglob('experiment.json')):
        run = json.loads(path.read_text())
        r, spec = run['recipe'], run['config']
        if r['kind'] != 'benchmark':
            continue
        valpath = path.parent / 'benchmark-val.json'
        val = json.loads(valpath.read_text()) if valpath.exists() else {}
        if val and not (run['status'] == 'trained' and run.get('completed_epochs') == r['epochs']
                        and run.get('effective_batch') == r['batch'] and run.get('best_sha256')
                        and run['best_sha256'] == val.get('weights_sha256')
                        and run['source_sha256'] == val.get('source_sha256')):
            raise ValueError(f'incomplete training or mismatched validation/weights/source: {path.parent}')
        latency_path = path.parent / 'latency.json'
        lat = json.loads(latency_path.read_text()) if latency_path.exists() else {}
        if lat and (not run.get('best_sha256') or lat.get('weights_sha256') != run['best_sha256'] or lat.get('imgsz') != r['imgsz']):
            raise ValueError(f'latency weights or resolution do not match this run: {latency_path}')
        lat_protocol = {k: lat.get(k) for k in ('hardware', 'device', 'cpu_threads', 'opencv_threads', 'torch', 'opencv', 'batch',
                        'precision', 'imgsz', 'rect', 'boundary', 'conf', 'iou', 'max_det', 'warmup', 'runs', 'images_sha256')}
        comparable = [run['base_commit'], run['source_sha256'], r, run.get('data_sha256'), run.get('init_sha256'),
                      run.get('hardware'), run.get('torch'), run.get('torchvision'), run.get('effective_batch'), run.get('optimizer_steps'),
                      run.get('train_split_sha256'), run.get('val_split_sha256'), val.get('split_sha256'),
                      val.get('images'), [b['gt'] for b in val.get('bins', {}).values()], val.get('data_sha256'),
                      val.get('split'), val.get('imgsz'), val.get('batch'), val.get('protocol'),
                      spec['options'], spec['order'], spec['tal_min']]
        group, bits = digest(comparable), bits_of(spec)
        bins = val.get('bins', {})
        row = {'group': group, 'id': bits, 'run': str(path.parent), 'commit': run['base_commit'],
               'source_sha256': run['source_sha256'], 'config': json.dumps(spec), **r,
               'protocol': json.dumps(val.get('protocol')), 'status': 'validated' if val else run['status'],
               'mAP50-95': val.get('mAP50-95'), **{f'R_{k}': bins.get(k, {}).get('recall') for k in ('lt8','8to16','ge16')},
               'effective_batch': run.get('effective_batch'), 'optimizer_steps': run.get('optimizer_steps'),
               'latency_group': digest(lat_protocol) if lat else None, 'latency_protocol': json.dumps(lat_protocol) if lat else None,
               'latency_hardware': lat.get('hardware'), 'p50_ms': lat.get('p50_ms'), 'p95_ms': lat.get('p95_ms'),
               'delta_mAP': None, 'interaction_mAP': None}
        rows.append(row)
        if val:
            cohort = groups.setdefault(group, {})
            if bits in cohort:
                raise ValueError(f'duplicate runs for {bits} in {group}; isolate one repeat per seed and recipe')
            cohort[bits] = row
    for cohort in groups.values():
        if '00000' not in cohort:
            continue
        base = cohort['00000']['mAP50-95']
        for bits, row in cohort.items():
            row['delta_mAP'] = row['mAP50-95'] - base
            singles = ['0'*i+'1'+'0'*(4-i) for i in range(5) if bits[i] == '1']
            if len(singles) >= 2 and all(s in cohort for s in singles):
                row['interaction_mAP'] = row['delta_mAP'] - sum(cohort[s]['mAP50-95'] - base for s in singles)
    if not rows:
        raise ValueError('no benchmark runs; smoke observations are not benchmark results')
    write_csv(output, rows)
