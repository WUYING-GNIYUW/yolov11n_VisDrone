import torch
from torch import nn

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import BaseModel, DetectionModel
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import dist2bbox, make_anchors

from .config import METHODS, normalize


def build_model(spec, nc=10, ch=3, verbose=False, weights=None, names=None):
    spec = normalize(spec)
    base = DetectionModel('yolo11n.yaml', ch=ch, nc=nc, verbose=verbose)
    base.args = get_cfg()
    if names:
        base.names = dict(names)
    if weights is not None:
        base.load(weights, verbose=verbose)
    if not any(spec[m] for m in METHODS) and spec['tal_min'] == 'auto':
        return base
    return ExperimentModel(base, spec)


def gated_detect(head, feats, mask, dense=False):
    b = feats[0].shape[0]
    active = torch.cat([mask[:, i:i+1].expand(b, f.shape[-2] * f.shape[-1]) for i, f in enumerate(feats)], 1)
    if dense:
        return {**head.forward_head(feats, **head.one2many), 'active': active}
    boxes, scores = [], []
    for i, f in enumerate(feats):
        rows = mask[:, i].nonzero().flatten()
        n = f.shape[-2] * f.shape[-1]
        box = f.new_zeros(b, 4 * head.reg_max, n)
        cls = f.new_full((b, head.nc, n), -64.)
        if rows.numel():
            selected = f.index_select(0, rows)
            box[rows] = head.cv2[i](selected).flatten(2)
            cls[rows] = head.cv3[i](selected).flatten(2)
        boxes.append(box)
        scores.append(cls)
    return {'boxes': torch.cat(boxes, 2), 'scores': torch.cat(scores, 2), 'active': active, 'feats': feats}


class ExperimentModel(DetectionModel):
    def __init__(self, base, spec):
        BaseModel.__init__(self)
        self.model, self.save = base.model, base.save
        self.yaml, self.names, self.args, self.inplace = base.yaml, base.names, base.args, base.inplace
        self.experiment = spec
        head = self.model[-1]
        channels = [m[0].conv.in_channels for m in head.cv2]
        strides = [int(x) for x in head.stride.tolist()]
        shallow = self.model[2].cv2.conv.out_channels
        opts = spec['options']
        self.esod = self.dm = self.query = self.rfla = self.ugs = None
        if spec['esod']:
            from .adapters.esod import ESODAdapter
            self.esod = ESODAdapter(shallow, opts.get('esod', {}))
        if spec['dm_efs']:
            from .adapters.dm_efs import DMEFSAdapter
            backbone_channels = [self.model[i].cv2.conv.out_channels for i in (2, 4, 6, 10)]
            self.dm = DMEFSAdapter(shallow, channels, strides, opts.get('dm_efs', {}), backbone_channels=backbone_channels)
            self.dm.extend_head(head)
            channels, strides = self.dm.channels, self.dm.strides
        if spec['querydet']:
            from .adapters.querydet import QueryDetAdapter
            self.query = QueryDetAdapter(channels, strides, opts.get('querydet', {}))
        if spec['rfla']:
            from .adapters.rfla import RFLAAdapter
            self.rfla = RFLAAdapter(strides, opts.get('rfla', {}))
        if spec['ugs']:
            from .adapters.ugs import UGSAdapter
            self.ugs = UGSAdapter(head.reg_max, opts.get('ugs', {}))
            self.ugs.initialize_head(head)
        self.stride = head.stride
        self.last_diagnostics = {}

    def _apply(self, fn):
        super()._apply(fn)
        self.stride = self.model[-1].stride
        self.criterion = None
        return self

    def init_criterion(self):
        from .loss import ExperimentLoss
        return ExperimentLoss(self)

    def geometry(self, raw):
        head = self.model[-1]
        anchors, stride = make_anchors(list(raw['feats']), head.stride, 0.5)
        if self.ugs is None:
            pred_boxes = dist2bbox(head.dfl(raw['boxes']).transpose(1, 2), anchors, xywh=False)
        else:
            pred_boxes = self.ugs.decode(raw['boxes'].transpose(1, 2), anchors)
        return anchors, stride, pred_boxes

    def _raw(self, img, batch=None):
        hw = tuple(img.shape[-2:])
        if min(hw) < 64 or any(s % 32 for s in hw):
            raise ValueError('experiment input must be at least 64px and a multiple of 32 on each side')
        x = self.model[2](self.model[1](self.model[0](img)))
        es = None
        if self.esod is not None:
            x, es = self.esod.split(x, hw, batch=batch)
            hw = es['hw']
        meta = {'esod_state': es, 'hw': hw, 'image_count': img.shape[0]}
        if not x.shape[0]:
            return {**meta, 'feats': []}
        shallow = x
        cache = [None, None, x]
        head = self.model[-1]
        for m in self.model[3:-1]:
            if m.f != -1:
                x = cache[m.f] if isinstance(m.f, int) else [x if j == -1 else cache[j] for j in m.f]
            x = m(x)
            cache.append(x if m.i in self.save else None)
        native = [cache[i] for i in (16, 19, 22)]
        if self.dm is not None:
            feats, level_mask, state = self.dm.route(shallow, native, backbone_feats=[cache[i] for i in (2, 4, 6, 10)])
            meta['dm_state'] = state
        else:
            feats = native
            level_mask = torch.ones((x.shape[0], len(feats)), dtype=torch.bool, device=x.device)
        feats = list(feats)
        raw = (self.query(head, feats, level_mask) if self.query is not None else
               gated_detect(head, feats, level_mask, dense=self.training or self.dm is None))
        if self.dm is not None:
            masks = raw['active'].split([f.shape[-2] * f.shape[-1] for f in feats], dim=1)
            meta['dm_state']['head_executed'] = torch.stack([m.any(1) for m in masks], 1)
        return {**raw, **meta, 'level_mask': level_mask}

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if profile or visualize or augment or embed:
            raise ValueError('experiment route supports standard detect inference, not TTA/profile/embed')
        raw = self._raw(x)
        if self.training:
            return raw
        if not raw['feats']:
            return x.new_zeros(x.shape[0], 4 + self.model[-1].nc, 0), raw
        _, stride, boxes = self.geometry(raw)
        xywh = xyxy2xywh(boxes * stride)
        valid = raw['active'] & (xywh[..., 2:] > 0).all(-1)
        scores = raw['scores'].sigmoid().masked_fill(~valid[:, None], 0)
        decoded = torch.cat((xywh.transpose(1, 2), scores), 1)
        if self.esod is not None:
            decoded = self.esod.restore(decoded, raw['esod_state'])
        return decoded, raw

    def loss(self, batch, preds=None):
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        raw = self._raw(batch['img'], batch if self.training else None) if preds is None else self.criterion.parse_output(preds)
        return self.criterion.compute(self, raw, batch)
