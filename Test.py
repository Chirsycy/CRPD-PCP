import numpy as np
import argparse
from torch.autograd import Variable
import scipy.io as io
import torch
import warnings

warnings.filterwarnings("ignore")


def getTestData_up2now(classes):
    test_dataset = io.loadmat('./dataset.mat')
    test_input, test_output = test_dataset['test_data'], test_dataset['test_labels']
    test_col, test_row = test_dataset['test_col'], test_dataset['test_row']

    datas, labels = [], []
    row, col = [], []
    for label in classes:
        for j in range(test_output.shape[1]):
            if test_output[0][j] == label:
                data = test_input[j]
                datas.append(data)
                labels.append(label)
                col.append(test_col[0][j])
                row.append(test_row[0][j])

    TestData, TestLabels = np.array(datas), np.array(labels)
    return TestData, TestLabels, col, row


def main(args):
    path = './results.txt'
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '_' + str(task_size)
    perm_id = [3, 9, 10, 11, 5, 13, 6, 4, 0, 14, 7, 2, 12, 1, 8]
    all_classes = np.array(perm_id)
    class_map = {val: idx for idx, val in enumerate(perm_id)}
    map_reverse = {v: k for k, v in class_map.items()}

    print("############# 初类vs初模、初类vs增模、增类vs增模 上的测试精度 #############")
    acc_all = []
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = f"{args.save_path}/{file_name}/{class_index}_model.pkl"
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        if current_task == 0:
            print('《初始模型》')
        else:
            print('《增量模型》')
        for i in range(current_task + 1):
            if i == 0:
                classes = all_classes[:args.fg_nc]
                print('初始类别')
            else:
                classes = all_classes[args.fg_nc + (i - 1) * task_size:args.fg_nc + i * task_size]
                print('增量类别')
            test_data, test_label, test_col, test_row = getTestData_up2now(classes)
            correct, total = 0, 0
            if torch.cuda.is_available():
                model.cuda()
            with torch.no_grad():
                for j in range(0, test_data.shape[0], args.batch_size):
                    imgs = torch.as_tensor(test_data[j:j + args.batch_size].transpose(0, 3, 1, 2), dtype=torch.float32)
                    labels = torch.as_tensor(test_label[j:j + args.batch_size], dtype=torch.float32)
                    if torch.cuda.is_available():
                        imgs, labels = imgs.cuda(), labels.cuda()
                    outputs = model(imgs)
                    predicts = torch.max(outputs, dim=1)[1]
                    predicts = torch.tensor([map_reverse[pred] for pred in predicts.cpu().numpy()])
                    correct += (predicts == labels.cpu()).sum().item()
                    total += len(labels)

            accuracy = correct / total
            print(f'精度: {accuracy:.4f}')
            acc_up2now.append(accuracy)
        acc_all.append(acc_up2now)
    flattened_acc = [item for sublist in acc_all for item in sublist]

    with open(path, 'a') as f:
        f.write("Initial OA==Incremental old OA==Incremental new OA")
        f.write('\n')
        for value in flattened_acc:
            f.write(str(value))
            f.write('\n')
        f.write('\n')

    print("############# 初类vs初模、全部类vs增模 #############")
    # 对当前任务进行迭代
    for current_task in range(args.task_num + 1):
        if current_task == 0:
            print('初始类别在初始模型上的精度')
        else:
            print('全部类别在增量模型上的精度')
        class_index = args.fg_nc + current_task * task_size
        filename = f"{args.save_path}{file_name}/{class_index}_model.pkl"
        model = torch.load(filename).to(device).eval()
        classes = all_classes[:args.fg_nc + current_task * task_size]
        test_data, test_label, ttest_col, ttest_row = getTestData_up2now(classes)
        correct, total = 0.0, 0.0
        y_pre = torch.tensor([0])
        n = 0

        for j in range(test_data.shape[0]):  # 样本数量进行迭代
            if (j + 1) % args.batch_size == 0:
                imgs, labels = test_data[n:j + 1].transpose((0, 3, 1, 2)), test_label[n:j + 1]
                n += args.batch_size
                imgs = torch.as_tensor(torch.from_numpy(imgs), dtype=torch.float32)
                labels = torch.as_tensor(torch.from_numpy(labels), dtype=torch.float32)
                if torch.cuda.is_available():
                    imgs, labels = imgs.cuda(), labels.cuda()
                with torch.no_grad():
                    outputs = model(imgs)
                predicts = torch.max(outputs, dim=1)[1]
                predicts = torch.tensor([map_reverse[pred] for pred in predicts.cpu().numpy()])
                correct += (predicts == labels.cpu()).sum()
                total += len(labels)
                y_pre = torch.cat([y_pre, predicts])

        if (test_data.shape[0] - n) < args.batch_size and (test_label.shape[0] - n) != 0:
            imgs, labels = test_data[n:].transpose((0, 3, 1, 2)), test_label[n:]
            imgs = torch.as_tensor(torch.from_numpy(imgs), dtype=torch.float32)
            labels = torch.as_tensor(torch.from_numpy(labels), dtype=torch.float32)
            if torch.cuda.is_available():
                imgs, labels = imgs.cuda(), labels.cuda()
            with torch.no_grad():
                outputs = model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            predicts = torch.tensor([map_reverse[pred] for pred in predicts.cpu().numpy()])
            correct += (predicts == labels.cpu()).sum()
            total += len(labels)
            y_pre = torch.cat([y_pre, predicts])
        import sklearn.metrics as metrics

        y_pre_list = [[] for _ in range(args.total_nc)]
        y_true_list = [[] for _ in range(args.total_nc)]

        for i, label in enumerate(test_label):
            y_true_list[label].append(test_label[i])
            y_pre_list[label].append(y_pre[i])

        class_OA = [metrics.accuracy_score(y_pre_list[i], y_true_list[i]) for i in range(len(y_true_list))]
        Kappa = metrics.cohen_kappa_score(y_pre[1:], test_label)

        print("kappa:", Kappa)
        accuracy = correct.item() / total
        print("accuracy", accuracy)

        with open(path, 'a') as f:
            f.write("kappa:\n")
            f.write(str(Kappa) + '\n')
            oa_type = "Initial" if current_task == 0 else "Incremental"
            f.write(f"{oa_type} OA:\n")
            f.write(str(accuracy) + '\n')
            for oo, oa in enumerate(class_OA):
                f.write(f"类别{oo + 1}: {oa}\n")

        if len(classes) == 15:
            new_output = np.zeros((349, 1905))
            yy = y_pre.numpy().tolist()

            for i, (col, row) in enumerate(zip(ttest_col, ttest_row)):
                new_output[col][row] = yy[i + 1] + 1

            yy = {'y_pre': new_output}
            io.savemat("./y_pre_houston_9.mat", yy)
