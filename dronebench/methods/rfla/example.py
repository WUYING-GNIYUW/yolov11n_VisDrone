import torch

from dronebench.methods.rfla import hierarchical_assign, receptive_field_scores


def main():
    points = torch.tensor([[3.0, 5.0], [7.0, 5.0], [13.0, 5.0]])
    radii = torch.tensor([1.0, 1 / 0.9, 2.0])
    boxes = torch.tensor([[4.0, 4.0, 6.0, 6.0]])
    print("第一轮 RFD:", receptive_field_scores(points, radii, boxes).tolist())
    result = hierarchical_assign(points, radii, boxes, topk=1, beta=0.9)
    print("GT 编号:", result.gt_index.tolist())
    print("来源轮次:", result.stage.tolist())
    print("每个 GT 最终正样本数:", result.counts.tolist())
    print("框外正样本:", result.outside_gt.tolist())


if __name__ == "__main__":
    main()
