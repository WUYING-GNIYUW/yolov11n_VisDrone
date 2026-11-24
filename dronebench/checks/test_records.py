import csv
import json
import shlex

import pytest

from dronebench.config import BASE, bits_of, from_id
from dronebench.records import matrix, run_name, summarize, write_json


def plan():
    return dict(data='dataset with space.yaml', epochs=100, batch=8, imgsz=640, seed=0, device='cpu', workers=0,
                init=None, kind='benchmark')


def test_matrix_has_unique_nonoverlapping_commands(tmp_path):
    path = tmp_path / 'queue.csv'
    matrix(plan(), 'project with space', path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len({r['run'] for r in rows}) == len(rows) == 32
    assert {r['id'] for r in rows} == {f'{i:05b}' for i in range(32)}
    for row in rows:
        spec = json.loads(row['config'])
        assert bits_of(spec) == row['id'] and row['mAP50-95'] == ''
        train, val = shlex.split(row['train_command']), shlex.split(row['val_command'])
        assert train[train.index('--data')+1] == plan()['data']
        assert val[val.index('--weights')+1] == 'project with space/' + row['run'] + '/weights/best.pt'
        assert row['run'] != run_name(spec, {**plan(), 'imgsz': 960})


def specimen(root, bits, score, recipe=None, label_hash='same'):
    r = recipe or plan()
    folder = root / (bits + label_hash + r['kind'])
    folder.mkdir()
    run = dict(base_commit=BASE, source_sha256='fixture', recipe=r, config=from_id(bits), status='trained',
               completed_epochs=r['epochs'], effective_batch=r['batch'], optimizer_steps=10, best_sha256=bits,
               train_split_sha256='train', val_split_sha256=label_hash)
    write_json(folder/'experiment.json', run)
    bins = {k: {'gt': 12, 'recall': 0.25} for k in ('lt8','8to16','ge16')}
    write_json(folder/'benchmark-val.json', {'weights_sha256': bits, 'source_sha256': 'fixture', 'split_sha256': label_hash,
                                              'mAP50-95': score, 'bins': bins, 'split':'val', 'imgsz':640})
    return folder


def test_collect_excludes_smoke_and_compares_only_matched_factorials(tmp_path):
    for bits, ap in [('00000', .2), ('00001', .25), ('00010', .24), ('00011', .26)]:
        specimen(tmp_path, bits, ap)
    specimen(tmp_path, '00000', .9, recipe={**plan(), 'kind': 'smoke'})
    specimen(tmp_path, '00011', .01, label_hash='different')
    out = tmp_path/'measured.csv'
    summarize(tmp_path, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    both, = [r for r in rows if r['id'] == '00011' and r['interaction_mAP']]
    assert float(both['delta_mAP']) == pytest.approx(.06)
    assert float(both['interaction_mAP']) == pytest.approx(-.03)
    other, = [r for r in rows if r['mAP50-95'] == '0.01']
    assert other['delta_mAP'] == ''


def test_collect_refuses_stale_validation_and_wrong_latency(tmp_path):
    folder = specimen(tmp_path, '00000', .1)
    runpath = folder/'experiment.json'
    run = json.loads(runpath.read_text())
    write_json(runpath, {**run, 'completed_epochs': 1})
    with pytest.raises(ValueError, match='incomplete'):
        summarize(tmp_path, tmp_path/'out.csv')
    write_json(runpath, run)
    write_json(folder/'latency.json', {'weights_sha256':'other', 'imgsz':640})
    with pytest.raises(ValueError, match='latency'):
        summarize(tmp_path, tmp_path/'out.csv')
