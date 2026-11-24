import torch
from torch import nn
from torch.nn import functional as F

from . import ObjSeeker, foreground_loss, foreground_targets, select_patches


def predict_patches(stem_features, layout, head, stride):
    raw = head(layout.crop(stem_features))
    ph, pw = layout.patch_hw
    extent = raw.new_tensor((pw, ph)) * stride
    center = raw[:, :2].sigmoid() * extent
    half = raw[:, 2:4].sigmoid() * extent / 2
    return raw[:, 4], torch.cat((center - half, center + half), dim=1)


def main():
    torch.manual_seed(7)
    stride = 2
    patch_hw = (8, 8)
    stem = nn.Sequential(nn.Conv2d(3, 8, 3, stride, 1), nn.ReLU())
    seeker = ObjSeeker(8)
    head = nn.Sequential(
        nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 5)
    )
    optimizer = torch.optim.SGD([*stem.parameters(), *seeker.parameters(), *head.parameters()], lr=0.01)
    image = torch.rand(2, 3, 32, 48)
    boxes = torch.tensor([[8.0, 8.0, 12.0, 12.0], [32.0, 20.0, 36.0, 24.0]])
    image_ids = torch.tensor([0, 1])

    features = stem(image)
    logits = seeker(features)
    target = foreground_targets(logits, boxes, image_ids, stride)
    layout = select_patches(target, patch_hw)
    loss = foreground_loss(logits, target)
    if len(layout.offsets):
        scores, local_pred = predict_patches(features, layout, head, stride)
        local_gt, patch_ids, _ = layout.project_boxes(boxes, image_ids, stride)
        labels = torch.zeros_like(scores)
        labels[patch_ids] = 1
        loss = loss + F.binary_cross_entropy_with_logits(scores, labels)
        if len(local_gt):
            loss = loss + F.smooth_l1_loss(local_pred[patch_ids], local_gt)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"warmup: {len(layout.offsets)} patches, loss={loss.item():.3f}")

    stem.eval(), seeker.eval(), head.eval()
    with torch.no_grad():
        features = stem(image)
        layout = select_patches(seeker(features).sigmoid(), patch_hw, threshold=0.4)
        if not len(layout.offsets):
            print("inference: 0 patches, no detections")
            return
        scores, local_pred = predict_patches(features, layout, head, stride)
        patch_ids = torch.arange(len(scores), device=scores.device)
        keep = layout.valid_centers((local_pred[:, :2] + local_pred[:, 2:]) / 2, patch_ids, stride)
        original, ids = layout.restore_boxes(local_pred[keep], patch_ids[keep], stride, image_hw=image.shape[-2:])
        print(f"inference: {len(layout.offsets)} patches; {len(original)} boxes before NMS; image ids={ids.unique().tolist()}")


if __name__ == "__main__":
    main()
