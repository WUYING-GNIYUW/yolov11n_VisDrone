import torch
from torch import nn

from . import QueryHead, cascade_plans, query_focal_loss, query_targets


def main():
    torch.manual_seed(7)
    coarse = torch.full((1, 1, 4, 5), -4.0)
    middle = torch.full((1, 1, 7, 9), -4.0)
    fine = torch.randn(1, 3, 14, 18)
    coarse[0, 0, 1, 2] = 4.0
    middle[0, 0, 2, 5] = 4.0
    middle[0, 0, 4, 7] = 4.0
    queries = nn.ModuleList(QueryHead(1, depth=0) for _ in range(2))
    with torch.no_grad():
        for head in queries:
            head[0].weight.zero_()
            head[0].weight[0, 0, 1, 1] = 1
    boxes = [torch.tensor([[19.0, 7.0, 25.0, 13.0]])]
    loss = torch.zeros(())
    for head, feature, stride, max_size in zip(queries, (coarse, middle), (8, 4), (32, 16)):
        logits = head(feature)
        target = query_targets(logits, boxes, stride, (0, max_size))
        loss = loss + query_focal_loss(logits, target)
    loss.backward()
    optimizer = torch.optim.SGD(queries.parameters(), lr=0.01)
    optimizer.step()
    with torch.no_grad():
        plans = cascade_plans((coarse, middle, fine), queries.eval(), threshold=0.5, radius=1)
        predictor = nn.Conv2d(3, 6, 3, padding=1).eval()
        predictions = plans[-1].run(predictor, fine)
    print(f"dense query training loss: {loss.item():.4f}")
    for level, plan in enumerate(plans, 1):
        b, h, w = plan.shape
        print(f"fine level {level}: outputs={len(plan.keys)}, computed sites={len(plan.active)}, grid={b * h * w}")
    print(f"final (batch, y, x): {plans[-1].keys.tolist()}, prediction shape: {tuple(predictions.shape)}")


if __name__ == "__main__":
    main()
