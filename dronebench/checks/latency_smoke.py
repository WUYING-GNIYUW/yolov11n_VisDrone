import argparse
import gc
import json
import shlex
import sys
from pathlib import Path

from dronebench.config import BASE, bits_of, combinations
from dronebench.evaluation.latency import benchmark
from dronebench.records import source_hash, write_json


def main():
    p = argparse.ArgumentParser(description='Exercise all 32 inference/latency paths with engineering-only checkpoints.')
    p.add_argument('--project', type=Path, required=True, help='completed dronebench.e2e_smoke project')
    p.add_argument('--images', type=Path, required=True, help='engineering smoke-val JPEG directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--warmup', default=3, type=int)
    p.add_argument('--runs', default=10, type=int)
    args = p.parse_args()
    receipts = []
    for spec in combinations():
        bits = bits_of(spec)
        path, = (args.project / bits).glob('*/experiment.json')
        receipt = json.loads(path.read_text())
        if (receipt['id'] != bits or receipt['recipe']['kind'] != 'smoke' or receipt['status'] != 'trained'
                or receipt['base_commit'] != BASE or receipt['source_sha256'] != source_hash()):
            raise ValueError(f'not a completed engineering run of this source: {path}')
        receipts.append((bits, receipt))
    out = {'scope': 'engineering latency smoke; undertrained checkpoints, two validation images, not benchmark results',
           'base_commit': BASE, 'source_sha256': source_hash(), 'rows': [],
           'command': shlex.join([sys.executable, '-m', 'dronebench.checks.latency_smoke', *sys.argv[1:]])}
    for bits, rec in receipts:
        m = benchmark(rec['best'], args.images, imgsz=rec['recipe']['imgsz'], warmup=args.warmup, runs=args.runs)
        if rec['best_sha256'] != m['weights_sha256']:
            raise ValueError(f'checkpoint differs: {bits}')
        k_per_run = {'weights', 'weights_sha256', 'samples', 'p50_ms', 'p95_ms'}
        protocol = {k: v for k, v in m.items() if k not in k_per_run}
        if out.setdefault('protocol', protocol) != protocol:
            raise ValueError('cannot combine records collected with different latency protocols')
        out['rows'].append({'id': bits, 'weights_sha256': m['weights_sha256'], 'p50_ms': m['p50_ms'], 'p95_ms': m['p95_ms'],
                            'samples': [{**s, 'file': Path(s['file']).name} for s in m['samples']]})
        print(f'{bits}: {m["p50_ms"]:.2f} / {m["p95_ms"]:.2f} ms', flush=True)
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, out)


if __name__ == '__main__':
    main()
