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


def cw(model, images, labels, c, kappa, max_iter, learning_rate, ii):
    # Define f-function
    def f(x):
        _, outputs = model(x, labels)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(images.device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction="sum")(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0:
            if cost > prev:
                print("Attack Stopped due to CONVERGENCE....")
                return a
            prev = cost

        print("- Learning Progress : %2.2f %%, Iteration: %d\t\t\t\t" %
              ((step+1)/max_iter*100, ii), end="\r")

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images
