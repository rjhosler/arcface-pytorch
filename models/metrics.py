from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
import math


class Softmax(nn.Module):
    def __init__(self, in_features, out_features):
        super(Softmax, self).__init__()
        self.layer = Linear(in_features, out_features)

    def forward(self, input, label):
        output = self.layer(input)
        return output


class AAML(nn.Module):
    r"""Center loss.
    
    Reference:
    Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device: torh device object cpu/gpu
        s: norm of input feature
        m: margin

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, device, s=30.0, m=0.50):
        super(AAML, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # add angular margin penalty to correct class during training only
        if self.training:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            cos_m = 1
            sin_m = 0
            th = 1
            mm = 0
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * cos_m - sine * sin_m
            phi = torch.where(cosine > th, phi, cosine - mm)
            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        return output


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes: number of classes.
        feat_dim: feature dimension.
        device: torh device object cpu/gpu
    """

    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(
            self.num_classes, self.feat_dim)).to(self.device)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def pairwise_distances(embeddings, squared=False, adv_embeddings=None):
    if adv_embeddings is not None:
        adv_embeddings = torch.nn.functional.normalize(adv_embeddings)
        embeddings = torch.nn.functional.normalize(embeddings)
        pdist_mat = adv_embeddings.mm(embeddings.t())

    else:
        embeddings = torch.nn.functional.normalize(embeddings)
        pdist_mat = embeddings.mm(embeddings.t())

    cos_similarity = 1 - pdist_mat.abs()

    return cos_similarity.clamp(min=0, max=1)


def get_valid_triplets_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
    indices_not_equal = ~indices_equal
    i_ne_j = indices_not_equal.unsqueeze(2)
    i_ne_k = indices_not_equal.unsqueeze(1)
    j_ne_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_ne_j & i_ne_k & j_ne_k

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    i_eq_j = label_equal.unsqueeze(2)
    i_eq_k = label_equal.unsqueeze(1)
    i_ne_k = ~i_eq_k
    valid_labels = i_eq_j & i_ne_k

    mask = distinct_indices & valid_labels
    return mask


def batch_all_triplet_loss(labels, embeddings, margin, squared=False, adv_embeddings=None):
    """
    get triplet loss for all valid triplets and average over those triplets whose loss is positive.
    """
    distances = pairwise_distances(
        embeddings, squared=squared, adv_embeddings=adv_embeddings)

    anchor_positive_dist = distances.unsqueeze(2)
    anchor_negative_dist = distances.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # get a 3D mask to filter out invalid triplets
    mask = get_valid_triplets_mask(labels)

    triplet_loss = triplet_loss * mask.float()
    triplet_loss.clamp_(min=0)

    # count the number of positive triplets
    epsilon = 1e-16
    num_positive_triplets = (triplet_loss > 0).float().sum()
    num_valid_triplets = mask.float().sum()
    fraction_positive_triplets = num_positive_triplets / \
        (num_valid_triplets + epsilon)

    triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

    return triplet_loss, fraction_positive_triplets, mask.cpu().numpy()


def get_valid_positive_mask(labels):
    """
    To be a valid positive pair (a,p),
        - a and p are different embeddings
        - a and p have the same label
    """
    indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
    indices_not_equal = ~indices_equal

    label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_equal
    return mask


def get_valid_negative_mask(labels):
    """
    To be a valid negative pair (a,n),
        - a and n are different embeddings
        - a and n have the different label
    """
    indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
    indices_not_equal = ~indices_equal

    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_not_equal
    return mask


def batch_all_contrastive_loss(labels, embeddings, margin, squared=False, adv_embeddings=None):
    """
    get contrastive loss for all valid pairs and average over those pairs whose loss is positive.
    """
    distances = pairwise_distances(
        embeddings, squared=squared, adv_embeddings=adv_embeddings)

    pos_mask = get_valid_positive_mask(labels)

    neg_mask = get_valid_negative_mask(labels)
    neg_mask = (margin * neg_mask.float()) - \
        (distances * neg_mask.float())
    neg_mask.clamp_(min=0)
    neg_mask = neg_mask > 0

    mask = pos_mask | neg_mask

    epsilon = 1e-16

    pos_distances = distances * pos_mask.float()
    pos_cnt = torch.sum(pos_distances > 0)
    pos_loss = pos_distances.sum() / (pos_cnt + epsilon)

    neg_distances = (margin * neg_mask.float()) - \
        (distances * neg_mask.float())
    neg_distances.clamp_(min=0)
    neg_cnt = torch.sum(neg_mask > 0)
    neg_loss = neg_distances.sum() / (neg_cnt + epsilon)

    contrastive_loss = pos_loss + neg_loss
    
    return contrastive_loss, mask.cpu().numpy()
