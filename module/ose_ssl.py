import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSEProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, features):
        return F.normalize(self.projector(features), dim=1)


class OSEMemoryBank(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.size = int(size)
        self.dim = int(dim)
        self.register_buffer("features", torch.zeros(self.size, self.dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    def reset(self):
        self.features.zero_()
        self.ptr.zero_()
        self.filled.zero_()

    @torch.no_grad()
    def enqueue(self, features):
        features = F.normalize(features.detach(), dim=1)
        batch_size = features.shape[0]
        if batch_size >= self.size:
            self.features.copy_(features[-self.size:])
            self.ptr.zero_()
            self.filled.fill_(self.size)
            return

        ptr = int(self.ptr.item())
        end = ptr + batch_size
        if end <= self.size:
            self.features[ptr:end].copy_(features)
        else:
            first = self.size - ptr
            self.features[ptr:].copy_(features[:first])
            self.features[: end - self.size].copy_(features[first:])

        self.ptr[0] = end % self.size
        self.filled[0] = min(self.size, int(self.filled.item()) + batch_size)

    def get(self):
        filled = int(self.filled.item())
        if filled == 0:
            return None
        return self.features[:filled]


def select_one_exemplar_per_class(labels, num_classes, seed):
    class_to_indices = {class_id: [] for class_id in range(num_classes)}
    for index, label in enumerate(labels):
        label = int(label)
        if label in class_to_indices:
            class_to_indices[label].append(index)

    rng = random.Random(seed)
    indices = []
    missing = []
    for class_id in range(num_classes):
        candidates = class_to_indices[class_id]
        if not candidates:
            missing.append(class_id)
            continue
        indices.append(rng.choice(candidates))

    if missing:
        raise ValueError(
            "Cannot build one-shot exemplars; missing classes: {}".format(missing)
        )
    return indices


def build_exemplar_guided_prototypes(
    exemplar_features,
    memory_features,
    topk=8,
    alpha=0.75,
):
    exemplar_features = F.normalize(exemplar_features, dim=1)
    memory_features = F.normalize(memory_features, dim=1)

    num_classes = exemplar_features.shape[0]
    memory_size = memory_features.shape[0]
    k = min(int(topk), memory_size)

    similarity = torch.mm(exemplar_features, memory_features.t())
    if num_classes > 1:
        max_other = []
        for class_id in range(num_classes):
            other = torch.cat(
                [similarity[:class_id], similarity[class_id + 1 :]],
                dim=0,
            )
            max_other.append(other.max(dim=0).values)
        max_other = torch.stack(max_other, dim=0)
    else:
        max_other = torch.zeros_like(similarity)

    scores = alpha * similarity - (1.0 - alpha) * max_other
    _, top_indices = torch.topk(scores, k=k, dim=1)

    prototypes = []
    for class_id in range(num_classes):
        neighbors = memory_features[top_indices[class_id]]
        components = torch.cat(
            [exemplar_features[class_id : class_id + 1], neighbors],
            dim=0,
        )
        weights = torch.softmax(
            torch.mv(components, exemplar_features[class_id]),
            dim=0,
        )
        prototype = torch.sum(weights[:, None] * components, dim=0)
        prototypes.append(prototype)

    return F.normalize(torch.stack(prototypes, dim=0), dim=1)


def cross_entropy_from_probs(target_probs, pred_log_probs):
    return -(target_probs * pred_log_probs).sum(dim=1).mean()


class OSELoss(nn.Module):
    def __init__(self, tau_s=0.1, tau_t=0.04):
        super().__init__()
        self.tau_s = tau_s
        self.tau_t = tau_t

    def prototype_probs(self, features, prototypes, tau):
        logits = torch.mm(F.normalize(features, dim=1), prototypes.t()) / tau
        return torch.softmax(logits, dim=1)

    def align_loss(self, student_features, teacher_features, prototypes):
        student_logits = torch.mm(F.normalize(student_features, dim=1), prototypes.t())
        student_log_probs = F.log_softmax(student_logits / self.tau_s, dim=1)
        with torch.no_grad():
            teacher_probs = self.prototype_probs(
                teacher_features,
                prototypes,
                self.tau_t,
            )
        return cross_entropy_from_probs(teacher_probs, student_log_probs)

    def dispersion_loss(self, prototypes):
        similarity = torch.mm(prototypes, prototypes.t()) / self.tau_s
        num_classes = prototypes.shape[0]
        off_diag = similarity[~torch.eye(num_classes, dtype=torch.bool, device=similarity.device)]
        return off_diag.mean()

    def mix_loss(
        self,
        mixed_features,
        first_features,
        second_features,
        prototypes,
        beta,
        permutation,
    ):
        mixed_logits = torch.mm(F.normalize(mixed_features, dim=1), prototypes.t())
        mixed_log_probs = F.log_softmax(mixed_logits / self.tau_s, dim=1)
        with torch.no_grad():
            first_probs = self.prototype_probs(first_features, prototypes, self.tau_t)
            second_probs = self.prototype_probs(second_features, prototypes, self.tau_t)
            target_probs = beta * first_probs + (1.0 - beta) * second_probs[permutation]
        mix_proto = cross_entropy_from_probs(target_probs, mixed_log_probs)

        instance_logits = torch.mm(
            F.normalize(mixed_features, dim=1),
            F.normalize(second_features.detach(), dim=1).t(),
        )
        instance_log_probs = F.log_softmax(instance_logits / self.tau_s, dim=1)
        batch_size = mixed_features.shape[0]
        labels = torch.eye(batch_size, device=mixed_features.device)
        permuted_labels = labels[permutation]
        mixed_labels = beta * labels + (1.0 - beta) * permuted_labels
        mix_instance = cross_entropy_from_probs(mixed_labels, instance_log_probs)
        return mix_proto, mix_instance


@torch.no_grad()
def copy_params(source, target):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(source_param.data)
        target_param.requires_grad = False
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
        target_buffer.data.copy_(source_buffer.data)


@torch.no_grad()
def ema_update(source, target, momentum):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(momentum).add_(source_param.data, alpha=1.0 - momentum)
    for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
        target_buffer.data.copy_(source_buffer.data)
