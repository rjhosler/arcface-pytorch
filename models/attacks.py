import torch
from torch.nn import CrossEntropyLoss


def fgsm(model, images, labels, eps, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images, labels)

    cost = loss(outputs, labels).to(device)
    model.zero_grad()
    cost.backward()

    grad = images.grad.data.sign()
    adv_images = images + eps * grad
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images


def bim(model, images, labels, eps, alpha, iters, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    for i in range(iters):
        images.requires_grad = True

        outputs = model(images, labels)

        cost = loss(outputs, labels).to(device)
        model.zero_grad()
        cost.backward()

        grad = images.grad.data.sign()
        adv_images = images + alpha * grad
        a = torch.clamp(images - eps, min=0)
        b = (adv_images >= a).float() * adv_images + (a > adv_images).float() * a
        c = (b > images + eps).float() * (images+eps) + (images+eps >= b).float() * b
        images = torch.clamp(c, max=1).detach()

    adv_images = images

    return adv_images


def pgd(model, images, labels, eps, alpha, iters, device):
    loss = CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True

        outputs = model(images, labels)

        cost = loss(outputs, labels).to(device)
        model.zero_grad()
        cost.backward()

        grad = images.grad.data.sign()
        adv_images = images + alpha * grad
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images
