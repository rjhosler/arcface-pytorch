import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import functools


np.random.seed(42)
torch.manual_seed(42)


def fgsm(model, images, labels, eps):
    loss = nn.CrossEntropyLoss()

    images.requires_grad = True

    _, outputs = model(images, labels)

    cost = loss(outputs, labels)
    model.zero_grad()
    cost.backward()

    grad = images.grad.data.sign()
    adv_images = images + eps * grad
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images


def bim(model, images, labels, eps, alpha, iters):
    loss = nn.CrossEntropyLoss()

    input_images = images.data

    for i in range(iters):
        images.requires_grad = True

        _, outputs = model(images, labels)

        cost = loss(outputs, labels)
        model.zero_grad()
        cost.backward()

        grad = images.grad.data.sign()
        adv_images = images + alpha * grad

        zeros = torch.zeros_like(images).to(images.device)
        ones = zeros + 1

        clip_1 = functools.reduce(
            torch.max, [zeros, input_images - eps, adv_images])
        clip_2 = functools.reduce(
            torch.min, [ones, input_images + eps, clip_1])
        images = torch.clamp(clip_2, 0, 1).detach()

    adv_images = images

    return adv_images


def mim(model, images, labels, eps, alpha, momemtum, iters):
    loss = nn.CrossEntropyLoss()

    input_images = images.data
    grad_prev = torch.zeros_like(images).to(images.device)

    for i in range(iters):
        images.requires_grad = True

        _, outputs = model(images, labels)

        cost = loss(outputs, labels)
        model.zero_grad()
        cost.backward()

        if i == 0:
            grad = images.grad.data / images.grad.data.norm(p=1)
        else:
            grad = momemtum * grad_prev + \
                images.grad.data / images.grad.data.norm(p=1)

        adv_images = images + alpha * grad.sign()

        zeros = torch.zeros_like(images).to(images.device)
        ones = zeros + 1

        clip_1 = functools.reduce(
            torch.max, [zeros, input_images - eps, adv_images])
        clip_2 = functools.reduce(
            torch.min, [ones, input_images + eps, clip_1])
        images = torch.clamp(clip_2, 0, 1).detach()

        grad_prev = grad

    adv_images = images

    return adv_images


def pgd(model, images, labels, eps, alpha, iters):
    loss = nn.CrossEntropyLoss()

    input_images = images.data
    images = images + \
        torch.FloatTensor(images.shape).uniform_(-eps, eps).to(images.device)

    for i in range(iters):
        images.requires_grad = True

        _, outputs = model(images, labels)

        cost = loss(outputs, labels)
        model.zero_grad()
        cost.backward()

        grad = images.grad.data.sign()
        adv_images = images + alpha * grad
        eta = torch.clamp(adv_images - input_images, min=-eps, max=eps)
        images = torch.clamp(input_images + eta, min=0, max=1).detach_()

    adv_images = images

    return adv_images
