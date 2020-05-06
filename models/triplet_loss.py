import torch


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


def get_valid_positive_mask(labels):
    """
    To be a valid positive pair (a,p),
        - a and p are different embeddings
        - a and p have the same label
    """
    indices_equal = torch.eye(labels.size(0)).byte()
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
    indices_equal = torch.eye(labels.size(0)).byte()
    indices_not_equal = ~indices_equal

    label_not_equal = torch.ne(labels.unsqueeze(1), labels.unsqueeze(0))

    mask = indices_not_equal & label_not_equal
    return mask


def get_valid_triplets_mask(labels):
    """
    To be valid, a triplet (a,p,n) has to satisfy:
        - a,p,n are distinct embeddings
        - a and p have the same label, while a and n have different label
    """
    indices_equal = torch.eye(labels.size(0)).bool().cuda()
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
