import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


class ExperimentLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        if model.experiment['tal_min'] != 'auto':
            self.assigner.stride_val = model.experiment['tal_min']
        names = ['box_loss', 'cls_loss', 'dfl_loss' if model.ugs is None else 'ugs_loc_loss']
        if model.ugs is not None:
            names.extend(['ugs_um_loss', 'ugs_ur_loss'])
        if model.dm is not None:
            names.append('size_loss')
        if model.query is not None:
            names.append('query_loss')
        if model.esod is not None:
            names.append('foreground_loss')
        self.loss_names = tuple(names)

    @torch.no_grad()
    def _tal(self, scores, boxes, anchors, labels, gt, valid, active):
        def assign(s, b, a, l, g, v):
            algo = self.assigner
            if s.shape[1] < algo.topk:
                algo = TaskAlignedAssigner(topk=s.shape[1], num_classes=self.nc, alpha=algo.alpha,
                                           beta=algo.beta, stride=algo.stride)
                algo.stride_val = self.assigner.stride_val
            return algo(s, b, a, l, g, v)[1:]
        if active.all():
            return assign(scores, boxes, anchors, labels, gt, valid)
        targets, soft = torch.zeros_like(boxes), torch.zeros_like(scores)
        fg = torch.zeros_like(active)
        indices = torch.zeros_like(active, dtype=torch.long)
        for i in range(scores.shape[0]):
            mask = active[i]
            if not mask.any():
                continue
            t, s, f, j = assign(scores[i:i+1, mask], boxes[i:i+1, mask], anchors[mask],
                                labels[i:i+1], gt[i:i+1], valid[i:i+1])
            targets[i, mask], soft[i, mask] = t[0].to(targets.dtype), s[0].to(soft.dtype)
            fg[i, mask], indices[i, mask] = f[0], j[0].long()
        return targets, soft, fg, indices

    def compute(self, model, raw, batch):
        es = raw['esod_state']
        items = {}
        diagnostics = {}
        if es is not None:
            items['foreground_loss'] = model.esod.aux_loss(batch, es)
            diagnostics['patch_offsets'] = es['layout'].offsets.detach()
        if not raw['feats']:
            zero = items['foreground_loss'] * 0
            for name in self.loss_names:
                items.setdefault(name, zero)
        else:
            local = batch if es is None else model.esod.local_batch(batch, es)
            active, hw = raw['active'], raw['hw']
            scores = raw['scores'].transpose(1, 2).contiguous()
            targets = torch.cat((local['batch_idx'].view(-1, 1), local['cls'].view(-1, 1), local['bboxes']), 1)
            gt = self.preprocess(targets.to(scores.device), active.shape[0], scores.new_tensor([hw[1], hw[0]] * 2))
            labels, gt = gt.split((1, 4), 2)
            valid = (gt[..., 2:] > gt[..., :2]).all(-1, keepdim=True)
            anchors, step, boxes = model.geometry(raw)
            points_px = anchors * step
            boxes_px = boxes.detach() * step
            if model.rfla is not None:
                target, soft, fg, _, diag = model.rfla.assign(points_px, step, labels, gt, valid, active, self.nc)
                diagnostics.update(diag)
            else:
                target, soft, fg, _ = self._tal(scores.detach().sigmoid(), boxes_px.to(gt.dtype), points_px,
                                               labels, gt, valid, active)
                diagnostics['tal_min_px'] = self.assigner.stride_val
            norm = soft.sum().clamp_min(1.)
            bce = self.bce(scores, soft.to(scores.dtype))
            if self.class_weights is not None:
                bce = bce * self.class_weights
            items['cls_loss'] = (bce * active[..., None]).sum() / norm * self.hyp.cls
            if model.ugs is None:
                box, loc = self.bbox_loss(raw['boxes'].transpose(1, 2).contiguous(), boxes, anchors,
                                          target / step, soft, norm, fg, scores.new_tensor(hw), step)
                items['dfl_loss'] = loc * self.hyp.dfl
            else:
                box, loc, um, ur, diag = model.ugs.losses(model.model[-1], raw, boxes, anchors, target / step,
                                                         soft, norm, fg, refine=model.training and torch.is_grad_enabled())
                items.update(ugs_loc_loss=loc * self.hyp.dfl, ugs_um_loss=um, ugs_ur_loss=ur)
                diagnostics.update(diag)
            items['box_loss'] = box * self.hyp.box
            if model.dm is not None:
                diagnostics['dm'] = {k: v.detach() for k, v in raw['dm_state'].items() if isinstance(v, torch.Tensor) and k in ('size_range', 'neck_requested', 'neck_executed', 'head_requested', 'head_enabled', 'head_executed')}
                items['size_loss'] = model.dm.aux_loss(raw['dm_state'], local, hw, boxes_px, raw['feats'], model.training)
            if model.query is not None:
                items['query_loss'] = model.query.aux_loss(raw, local, hw)
            diagnostics.update(positive=fg.sum().detach(), active=active.sum(1).detach(),
                               level_mask=raw['level_mask'].detach())
        model.last_diagnostics = diagnostics
        values = torch.stack([items[k].reshape(()) for k in self.loss_names])
        return values * raw['image_count'], dict(zip(self.loss_names, values.detach()))
