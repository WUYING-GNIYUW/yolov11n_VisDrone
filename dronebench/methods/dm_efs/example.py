import torch
from torch import nn
from torch.nn import functional as F

from . import (
    ExpandedPyramid,
    SizeFeatureCodebook,
    SizeRangePredictor,
    expand_features,
    paired_box_ious,
    predicted_size_range,
    run_selected,
    size_range_loss,
)


def main():
    torch.manual_seed(7)
    images = torch.zeros(2, 1, 32, 32)
    images[0, :, 14:18, 14:18] = 1
    images[1, :, 8:24, 8:24] = 1
    boxes = [torch.tensor([[14.0, 14, 18, 18]]), torch.tensor([[8.0, 8, 24, 24]])]
    stages = nn.ModuleList(nn.Conv2d(c, 4, 3, stride=2, padding=1) for c in (1, 4, 4))
    neck = ExpandedPyramid([4, 4, 4], width=4)
    controller = SizeRangePredictor([4, 4, 4], max_size=32)
    book = SizeFeatureCodebook(num_levels=3, max_size=32)
    heads = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 4)) for _ in stages)
    with torch.no_grad():
        for head, side in zip(heads, (4, 8, 16)):
            low = (32 - side) / 2
            head[-1].bias.copy_(torch.tensor([low, low, low + side, low + side]))

    def backbone():
        x, levels = images, []
        for stage in stages:
            x = stage(x).relu()
            levels.append(x)
        return expand_features(levels[:1], levels[1:])

    model = nn.ModuleList([stages, neck, heads, controller, book])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    features = backbone()
    preds = torch.stack([head(x) for head, x in zip(heads, neck(features))], dim=1)
    det_loss = F.smooth_l1_loss(preds, torch.cat(boxes)[:, None].expand_as(preds))
    aux_loss = size_range_loss(controller(features), boxes)
    for gt, matched in zip(boxes, preds.detach()):
        book.observe(gt, paired_box_ious(gt, matched.unsqueeze(0)))
    (det_loss + aux_loss).backward()
    optimizer.step()
    print("training votes (size 4, size 16):", book.votes[:, [4 - 2, 16 - 2]].tolist())

    model.eval()
    with torch.no_grad():
        features = backbone()
        sizes = predicted_size_range(controller(features))
        neck_mask, head_mask = book.gates(sizes)
        for row in range(len(images)):
            last = int((neck_mask[row] | head_mask[row]).nonzero()[-1].item())
            local = neck([x[row : row + 1] for x in features], last_level=last)
            out = run_selected(local, head_mask[row : row + 1, : last + 1], heads[: last + 1])
            print(f"image {row}: predicted endpoints={sizes[row].tolist()}, executed head levels={list(out)}")


if __name__ == "__main__":
    main()
