def iou(pred, target):
    import torch
    ious = []
    prdLabel = torch.argmax(pred, dim = 1)
    tarLabel = torch.argmax(target, dim = 1)
    size, n_class, h, w = pred.shape
    print(pred.shape)
    print(target.shape)
    for cls in range(n_class):
        prdEqCls = torch.eq(prdLabel, cls).float() 
        tarEqCls = torch.eq(tarLabel, cls).float()   
        intersection = torch.sum(torch.mul(prdEqCls, tarEqCls)).item()
        union = intersection + torch.sum(torch.mul(prdEqCls, torch.add(torch.mul(tarEqCls, -1), 1))).item() + torch.sum(torch.mul(tarEqCls, torch.add(torch.mul(prdEqCls, -1), 1))).item()
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return ious


def pixel_acc(pred, target):
    # Pixel_acc is given by total correct predictions / total number of samples
    import torch
    prdLabel = torch.argmax(pred, dim = 1)
    tarLabel = torch.argmax(target, dim = 1)
    equals = torch.eq(prdLabel, tarLabel).float()
    return torch.mean(equals).item()