import json

import torch
from torch import nn

from . import decode_logits, decode_ltrb, encode_ltrb, localization_ce, make_support, refinement_loss, uncertainty_entropy


def run_example():
    torch.manual_seed(7)
    support = make_support(num_bins=9, alpha=4, beta=0.5)
    features = (torch.randn(1, 3, 1, 2, requires_grad=True),)
    head = nn.Conv2d(3, 4 * support.numel(), kernel_size=1)
    with torch.no_grad():
        head.bias.copy_((-(support - 1).square()).repeat(4))
    optimizer = torch.optim.SGD(head.parameters(), lr=0.05)
    points_grid = torch.tensor([[0.5, 0.5], [1.5, 0.5]])
    stride = torch.tensor([[8.0], [8.0]])
    boxes_px = torch.tensor([[[8.0, 0.0, 16.0, 8.0], [8.0, 0.0, 16.0, 8.0]]])
    target = encode_ltrb(points_grid, boxes_px / stride)

    def predict(fs):
        projected = head(fs[0]).permute(0, 2, 3, 1)
        return projected.reshape(projected.shape[0], -1, 4, support.numel())

    logits = predict(features)
    ce = localization_ce(logits, target, support).mean()
    um = uncertainty_entropy(logits).mean()
    ur = refinement_loss(features, um, lambda fs: uncertainty_entropy(predict(fs)).mean(), rho=0.5)
    loss = ce + 0.5 * um + 0.5 * ur
    optimizer.zero_grad()
    loss.backward()
    grad_norm = head.weight.grad.norm().item()
    optimizer.step()

    head.eval()
    with torch.no_grad():
        distances = decode_logits(predict(features), support)
        predicted_px = decode_ltrb(points_grid, distances) * stride
    return {
        "clean_ce": ce.item(),
        "clean_um": um.item(),
        "perturbed_um": ur.item(),
        "head_gradient_norm": grad_norm,
        "target_ltrb_grid": target.tolist(),
        "predicted_xyxy_px": predicted_px.tolist(),
    }


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
