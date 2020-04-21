import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize


def fgsm(model, metric_fc, images, labels, eps, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    features = model(images)
    outputs = metric_fc(features, labels)

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    grad = normalize(images.grad.sign())
    attack_images = images + eps * grad
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images


def bim(model, metric_fc, images, labels, eps, alpha, iters, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    for i in range(iters):
        images.requires_grad = True

        features = model(images)
        outputs = metric_fc(features, labels)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        grad = normalize(images.grad.sign())
        adv_images = images + alpha * grad
        a = torch.clamp(images - eps, min=0)
        b = (adv_images >= a).float() * \
            adv_images + (a > adv_images).float() * a
        c = (b > images + eps).float() * (images+eps) + \
            (images+eps >= b).float() * b
        images = torch.clamp(c, max=1).detach()

    adv_images = images

    return adv_images


def pgd(model, metric_fc, images, labels, eps, alpha, iters, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True

        features = model(images)
        outputs = metric_fc(features, labels)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        grad = normalize(images.grad.sign())
        adv_images = images + alpha * grad
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images
