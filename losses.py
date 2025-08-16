import torch
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
import torch.nn as nn
from utils import pseuable
import torch.nn.functional as F


def Loss_distribution(self, images, target, old_classes, new_classes):
    _, _, prototypes_group_first, _, _, _, _ = pseuable(self, images, target,
                                                        old_classes,
                                                        new_classes)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.ToTensor()
    ])

    augmented_images = torch.zeros_like(images)
    for i in range(images.size(0)):
        for j in range(images.size(1)):
            augmented_images[i, j] = transform(images[i, j])
    _, _, prototypes_group_second, _, _, _, _ = pseuable(self, augmented_images, target, old_classes,
                                                         new_classes)
    cos_sim = cosine_similarity(nn.Parameter(torch.from_numpy(prototypes_group_first)),
                                nn.Parameter(torch.from_numpy(prototypes_group_second)), dim=1)
    loss_distribution = 1 - cos_sim.mean()
    return loss_distribution


def cross_entropy_loss(self, imgs, target):
    target = torch.as_tensor(target, dtype=torch.long)
    target = target.cuda()
    output = self.model(imgs)
    output = output.cuda()
    self.args.temp = torch.tensor([1]).cuda()  # 1
    loss_cls = nn.CrossEntropyLoss()(output / self.args.temp, target)
    return loss_cls


def Loss_relationship(self, old_logits, new_logits, old_classes):
    T = 0.1
    origin_logits, new_logits = old_logits / self.args.temperature, new_logits / self.args.temperature
    old_logits = old_logits.detach()
    old_outputs = old_logits
    preds = F.softmax(old_outputs / T, dim=-1) + 1e-8
    pseudo_logits = new_logits[:, old_classes]
    pseudo_preds = F.softmax(pseudo_logits / T, dim=-1) + 1e-8
    weight = torch.sum(pseudo_logits, dim=-1, keepdim=True)
    weight = weight / torch.mean(weight)
    kl_div = F.kl_div(F.log_softmax(preds, dim=1), F.softmax(pseudo_preds, dim=1), reduction='batchmean')
    weighted_kl_div = torch.sum(kl_div * weight) / preds.size(0)
    return weighted_kl_div
