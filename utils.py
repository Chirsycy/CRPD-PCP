import datetime
import random

import numpy as np
import torch
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from sklearn.mixture import GaussianMixture


# 生成伪标签
def pseuable(self, images, target, old_classes, new_classes):
    indices_old = np.where(np.isin(target, old_classes))[0]
    images_old = images[indices_old].to(torch.float32).cuda()
    target_old = target[indices_old]

    np.random.seed(42)

    num_samples_old = len(target_old)
    num_samples_to_extract = int(0.1 * num_samples_old)

    random_indices = np.random.choice(num_samples_old, num_samples_to_extract, replace=False)
    images_old = images_old[random_indices]
    target_old = target_old[random_indices]
    indices_new = np.where(np.isin(target, new_classes))[0]
    images_new = images[indices_new].to(torch.float32).cuda()
    target_new = target[indices_new]
    len_proto_new = len(np.unique(target_new))
    n = 0
    features = []
    for i in range(images_new.shape[0]):
        if (i + 1) % self.args.batch_size == 0:
            images, target = images_new[n:i + 1], images_new[n:i + 1]
            images = torch.as_tensor(images, dtype=torch.float32).to(self.device)
            n += self.args.batch_size
            feature = self.old_new_model.feature(images)
            features.append(feature)
    if (images_new.shape[0] - n) < self.args.batch_size and (images_new.shape[0] - n) != 0:
        images, target = images_new[n:], images_new[n:]
        images = torch.as_tensor(images, dtype=torch.float32).to(self.device)
        feature = self.old_new_model.feature(images)
        features.append(feature)
    features = torch.vstack(features)
    features = features.cpu().detach().numpy()
    # 原型生成
    if features.shape[0] > len_proto_new + 10:
        num_proto = features.shape[0] - 5
    elif features.shape[0] == len_proto_new:
        num_proto = len_proto_new
    else:
        num_proto = features.shape[0] - 3
    prototypes, proto = Prototype(features, num_proto)
    num_proto_group = len_proto_new
    prototypes_group, proto_group = Prototype(prototypes, num_proto_group)
    pes_label, merge = merge_and_create_result_array(proto, proto_group)
    for proto_idx, feature_idxs in merge.items():
        if len(feature_idxs) <= 2:
            merge[proto_idx] = feature_idxs
            continue

        feature_vectors = features[feature_idxs]

        proto_feature = prototypes_group[proto_idx]

        sam_distances = [cosine(feature_vector, proto_feature) for feature_vector in feature_vectors]

        sorted_idxs = np.argsort(sam_distances)
        keep_idxs = sorted_idxs[:int(len(sorted_idxs) * 0.25)]

        merge[proto_idx] = [feature_idxs[idx] for idx in keep_idxs]
    all_feature_idxs = []
    for feature_idxs in merge.values():
        all_feature_idxs.extend(feature_idxs)

    targets_new = target_new[all_feature_idxs]
    images_new = images_new[all_feature_idxs]

    feature_final = features[all_feature_idxs]
    element, feature_group = calculate_feature_mean(targets_new, feature_final)

    images_final = (images_old, images_new)
    images_final = torch.cat(images_final, dim=0)
    targets_final = (target_old, targets_new)
    targets_final = np.concatenate(targets_final)
    return images_final, targets_final, prototypes_group, element, feature_group, images_new, targets_new


def Prototype(features, num_prototypes):
    random.seed(datetime.datetime.now().year)
    gmm = GaussianMixture(n_components=num_prototypes)
    gmm.fit(features)
    prototypes = gmm.means_

    def calculate_distance(feature, prototype):
        dot_product = np.dot(feature, prototype)
        feature_norm = np.linalg.norm(feature)
        prototype_norm = np.linalg.norm(prototype)
        sam_distance = np.arccos(dot_product / (feature_norm * prototype_norm))
        return sam_distance

    distances = np.zeros((len(features), num_prototypes))
    for i, feature in enumerate(features):
        for j, prototype in enumerate(prototypes):
            distances[i, j] = calculate_distance(feature, prototype)
    proto = {i: [] for i in range(num_prototypes)}
    for feature_idx in range(len(features)):
        nearest_prototype = np.argmin(distances[feature_idx])
        proto[nearest_prototype].append(feature_idx)
    return prototypes, proto


def merge_and_create_result_array(dict_a, dict_b):
    dict_c = {}
    for key, values in dict_b.items():
        new_values = []
        for value in values:
            if value in dict_a:
                new_values.extend(dict_a[value])
        dict_c[key] = sorted(set(new_values))

    total_length = sum(len(values) for values in dict_c.values())
    result_array = [0] * total_length

    for key, values in dict_c.items():
        for value in values:
            result_array[value] = key

    return result_array, dict_c


def calculate_accuracy(arr1, arr2):
    unique_elements, unique_counts = np.unique(arr1, return_counts=True)

    accuracies = {}
    total_accuracy = 0
    for element, count in zip(unique_elements, unique_counts):
        indices = np.where(arr1 == element)[0]

        max_count = 0
        for index in indices:
            curr_count = np.sum(arr2[index] == arr2[indices])
            if curr_count > max_count:
                max_count = curr_count

        accuracy = max_count / count
        accuracies[element] = accuracy
        total_accuracy += accuracy

    total_accuracy /= len(unique_elements)
    return total_accuracy, accuracies


def calculate_feature_mean(arr_a, features):
    unique_elements = np.unique(arr_a)

    unique_features = []

    for element in unique_elements:
        indices = np.where(arr_a == element)[0]

        element_features = features[indices]

        mean_feature = np.mean(element_features, axis=0)

        unique_features.append(mean_feature)

    unique_features = np.array(unique_features)

    return unique_elements, unique_features


def new_old_data(images, targets_final, element, feature_group, element_old, feature_group_old):
    c = np.intersect1d(element, element_old)

    D = feature_group[np.isin(element, c)]
    E = feature_group_old[np.isin(element_old, c)]
    similarities = np.zeros((D.shape[0], E.shape[0]))
    for i in range(D.shape[0]):
        for j in range(E.shape[0]):
            a_normalized = D[i] / norm(D[i])
            b_normalized = E[j] / norm(E[j])
            cosine_similarity = np.dot(a_normalized, b_normalized)
            angle = np.arccos(cosine_similarity)
            similarities[i, j] = angle

    result = []
    matching_labels = []
    for i, label_d in enumerate(element[np.isin(element, c)]):
        label_e = element_old[np.isin(element_old, c)][np.argmin(similarities[i])]
        if label_d == label_e:
            matching_labels.append(label_d)
        result.append(
            f"feature_group中标签值为{label_d}的特征与feature_group_old中标签值为{label_e}的特征最相似")
    result = []
    matching_labels = []
    for i, label_d in enumerate(element[np.isin(element, c)]):
        label_e = element_old[np.isin(element_old, c)][np.argmax(similarities[i])]
        if label_d == label_e:
            matching_labels.append(label_d)
        result.append(
            f"feature_group中标签值为{label_d}的特征与feature_group_old中标签值为{label_e}的特征最相似")
    old_label = matching_labels
    new_label = np.setdiff1d(element, element_old)

    mask_new = np.isin(targets_final, new_label)
    image_new = images[mask_new]
    target_new = targets_final[mask_new]

    mask_old = np.isin(targets_final, old_label)
    image_old = images[mask_old]
    target_old = targets_final[mask_old]
    return image_old, image_new, target_old, target_new
