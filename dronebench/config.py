import hashlib
import itertools
import json
from pathlib import Path

METHODS = ('dm_efs', 'esod', 'querydet', 'rfla', 'ugs')
ORDER = 'stem-block-pyramid-level-query-decode-restore-nms'
BASE = '55989f76c1e85c192ba12cb3e533e3de842a2301'


def from_id(bits):
    if not isinstance(bits, str) or len(bits) != len(METHODS) or set(bits) - {'0', '1'}:
        raise ValueError('id must be five bits in dm_efs,esod,querydet,rfla,ugs order')
    return normalize(dict(zip(METHODS, (b == '1' for b in bits))))


def normalize(spec):
    if not isinstance(spec, dict) or set(spec) - {*METHODS, 'options', 'tal_min', 'order'}:
        raise ValueError('unknown keys or invalid experiment config')
    flags = {m: spec.get(m, False) for m in METHODS}
    if any(type(v) is not bool for v in flags.values()):
        raise ValueError('method switches must be booleans')
    opts = spec.get('options', {})
    if not isinstance(opts, dict) or set(opts) - set(METHODS) or any(not isinstance(v, dict) for v in opts.values()):
        raise ValueError('options must contain known method names and option dictionaries')
    if any(opts.get(m) and not flags[m] for m in METHODS):
        raise ValueError('options on a disabled method would have no effect')
    floor = spec.get('tal_min', 'auto')
    if floor != 'auto' and (type(floor) not in (int, float) or not 0 < floor < float('inf')):
        raise ValueError('tal_min must be auto or a positive finite number of input pixels')
    if flags['rfla'] and floor != 'auto':
        raise ValueError('RFLA does not use a TAL floor')
    if spec.get('order', ORDER) != ORDER:
        raise ValueError(f'only tested order: {ORDER}')
    return {**flags, 'tal_min': floor, 'options': {k: v for k, v in opts.items() if v}, 'order': ORDER}


def bits_of(spec):
    return ''.join(str(int(spec[m])) for m in METHODS)


def read_spec(bits=None, config=None):
    return normalize(json.loads(Path(config).read_text())) if config else from_id(bits or '00000')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:10]


def combinations():
    return (from_id(''.join(parts)) for parts in itertools.product('01', repeat=5))
