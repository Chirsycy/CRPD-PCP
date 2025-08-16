import warnings

import numpy as np
import torch
import torch.utils.data

from feature_extraction import resnet18_cbam
from model import protoAugSSL

warnings.filterwarnings("ignore")


def main(args):
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '_' + str(task_size)
    feature_extractor = resnet18_cbam()
    perm_id = [3, 9, 10, 11, 5, 13, 6, 4, 0, 14, 7, 2, 12, 1, 8]
    all_classes = np.array(perm_id)
    class_map = {val: idx for idx, val in enumerate(perm_id)}
    map_reverse = {v: k for k, v in class_map.items()}
    model = protoAugSSL(args, all_classes, map_reverse, class_map, file_name, feature_extractor, task_size, device)
    for current_task in range(args.task_num + 1):
        if current_task == 0:
            print('#######################初次训练阶段###################')
        else:
            print('#######################增量学习阶段###################')
        model.beforeTrain(current_task)
        model.train(class_map, current_task)
        model.afterTrain()
