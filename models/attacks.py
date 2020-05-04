import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


np.random.seed(42)
torch.manual_seed(42)


def fgsm(model, images, labels, eps, device):
    loss = nn.CrossEntropyLoss()

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
    loss = nn.CrossEntropyLoss()

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
        b = (adv_images >= a).float() * \
            adv_images + (a > adv_images).float() * a
        c = (b > images + eps).float() * (images+eps) + \
            (images+eps >= b).float() * b
        images = torch.clamp(c, max=1).detach()

    adv_images = images

    return adv_images


def mim(model, images, labels, eps, alpha, momemtum, iters, device):
    loss = nn.CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)
    grad_prev = torch.zeros_like(images)

    for i in range(iters):
        images.requires_grad = True

        outputs = model(images, labels)

        cost = loss(outputs, labels).to(device)
        model.zero_grad()
        cost.backward()

        if i == 0:
            grad = images.grad.data
        else:
            grad = momemtum * grad_prev + (1 - momemtum) * images.grad.data
        adv_images = images + alpha * grad.sign()
        a = torch.clamp(images - eps, min=0)
        b = (adv_images >= a).float() * \
            adv_images + (a > adv_images).float() * a
        c = (b > images + eps).float() * (images+eps) + \
            (images+eps >= b).float() * b
        images = torch.clamp(c, max=1).detach()
        grad_prev = grad

    adv_images = images

    return adv_images


def pgd(model, images, labels, eps, alpha, iters, device):
    loss = nn.CrossEntropyLoss()

    images = images.to(device)
    labels = labels.to(device)

    ori_images = images.data
    images = images + \
        torch.FloatTensor(images.shape).uniform_(-eps, eps).to(device)

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


def cw(model, images, labels, c, kappa, max_iter, learning_rate, device, ii):

    images = images.to(device)
    labels = labels.to(device)

    # Define f-function
    def f(x):
        outputs = model(x, labels)
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        return torch.clamp(j-i, min=-kappa)

    w = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = optim.Adam([w], lr=learning_rate)

    prev = 1e10

    for step in range(max_iter):

        a = 1/2*(nn.Tanh()(w) + 1)

        loss1 = nn.MSELoss(reduction='sum')(a, images)
        loss2 = torch.sum(c*f(a))

        cost = loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0:
            if cost > prev:
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost

        print('- Learning Progress : %2.2f %%, Iteration: %d        ' %
              ((step+1)/max_iter*100, ii), end='\r')

    attack_images = 1/2*(nn.Tanh()(w) + 1)

    return attack_images
