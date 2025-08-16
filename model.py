import os
import random

import numpy as np
import scipy.io as io
import torch
from torch.autograd import Variable

from classifier import network
from losses import Loss_distribution, cross_entropy_loss, Loss_relationship
from utils import pseuable, calculate_feature_mean


class protoAugSSL:
    def __init__(self, args, all_classes, map_reverse, class_map, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.all_classes = all_classes
        self.map_reverse = map_reverse
        self.class_map = class_map
        self.epochs = args.epochs1
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.std = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.old_new_model = None
        self.margin = 20
        self.sc_T = 1
        self.kd_T = 2

        self.alpha = 1
        self.lloss = 0

        self.train_dataset = io.loadmat('./dataset.mat')
        self.test_dataset = io.loadmat('./dataset.mat')
        self.train_old_data = None
        self.train_old_label = None
        self.train_new_data = None
        self.train_new_label = None
        self.train_total_data = None
        self.train_total_label = None
        self.test_data = []
        self.test_label = []
        self.old_pseudo_data = None
        self.old_pseudo_label = None
        self.a = 0
        self.element_old = []
        self.feature_group_old = []

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task > 0:
            self.a = 1
            self.model.Incremental_learning(self.numclass)
            self.old_new_model.to(self.device)
            self.old_new_model.eval()
            self.old_new_model.Incremental_learning(self.numclass)
            self.old_new_model.cuda()

        self.model.train()
        self.model.cuda()

    def train(self, class_map, current_task):
        if current_task == 0:
            self.epochs = self.args.epochs1
        else:
            self.epochs = self.args.epochs2
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-5)
        if current_task == 0:
            old_classes = self.all_classes[:self.numclass]
            self.train_old_data, self.train_old_label = self.getTrainData(old_classes)
            self.train_old_data, self.train_old_label = self.random_sample(self.train_old_data, self.train_old_label)
            self.getTestData(old_classes)

            for epoch in range(1, self.epochs):
                n = 0
                for i in range(self.train_old_data.shape[0]):
                    if (i + 1) % self.args.batch_size == 0:
                        images, target = self.train_old_data[n:i + 1].transpose((0, 3, 1, 2)), self.train_old_label[
                                                                                               n:i + 1]
                        n += self.args.batch_size
                        images = torch.as_tensor(torch.from_numpy(images), dtype=torch.float32)
                        target = Variable(torch.LongTensor([class_map[label] for label in target]))

                        if torch.cuda.is_available():
                            images, target = images.cuda(), target.cuda()
                        images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                        images = images.view(-1, 20, 9, 9)
                        target = torch.stack([target for _ in range(4)], 1).view(-1)

                        opt.zero_grad()
                        loss = cross_entropy_loss(self, images, target)
                        self.lloss = loss.item()
                        loss.backward()
                        opt.step()

                if (self.train_old_data.shape[0] - n) < self.args.batch_size and (
                        self.train_old_data.shape[0] - n) != 0:
                    images, target = self.train_old_data[n:].transpose((0, 3, 1, 2)), self.train_old_label[n:]
                    images = torch.as_tensor(Variable(torch.from_numpy(images)), dtype=torch.float32)
                    target = Variable(torch.LongTensor([class_map[label] for label in target]))

                    if torch.cuda.is_available():
                        images, target = images.cuda(), target.cuda()
                    images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, 20, 9, 9)
                    target = torch.stack([target for _ in range(4)], 1).view(-1)
                    loss = cross_entropy_loss(self, images, target)
                    self.lloss = loss.item()
                    loss.backward()
                    opt.step()

                if epoch % self.args.print_freq == 0:
                    accuracy = self._test()
                    print('epoch:%d, accuracy:%.5f, loss:%.5f' % (epoch, accuracy, self.lloss))
            self.model.cpu()
            images_old = torch.as_tensor(torch.from_numpy(self.train_old_data.transpose((0, 3, 1, 2))),
                                         dtype=torch.float32)
            target_old = self.train_old_label
            feature_old = self.model.feature(images_old)
            element_old, feature_group_old = calculate_feature_mean(target_old, feature_old.detach().numpy())
            self.element_old = element_old
            self.feature_group_old = feature_group_old
        else:
            total_classes = self.all_classes[: self.args.fg_nc + current_task * self.task_size]  # 旧类和新类结合
            old_classes = self.all_classes[:self.args.fg_nc]
            new_classes = self.all_classes[self.args.fg_nc + (
                    current_task - 1) * self.task_size: self.args.fg_nc + current_task * self.task_size]
            self.train_total_data, self.train_total_label = self.getTrainData(total_classes)
            self.getTestData(total_classes)

            old_classes = Variable(torch.LongTensor([class_map[label] for label in old_classes]))
            new_classes = Variable(torch.LongTensor([class_map[label] for label in new_classes]))
            for epoch in range(1, self.epochs):
                n = 0
                for i in range(self.train_total_data.shape[0]):
                    if (i + 1) % self.args.batch_size == 0:
                        images, target = self.train_total_data[n:i + 1].transpose((0, 3, 1, 2)), self.train_total_label[
                                                                                                 n:i + 1]
                        n += self.args.batch_size
                        target = Variable(torch.LongTensor([class_map[label] for label in target]))
                        images = torch.as_tensor(images, dtype=torch.float32).to(self.device)

                        loss_distribution = Loss_distribution(self, images, target, old_classes, new_classes)
                        images, targets_final, _, element, feature_group, images_new, targets_new = pseuable(self,
                                                                                                             images,
                                                                                                             target,
                                                                                                             old_classes,
                                                                                                             new_classes)

                        loss_cross_entropy = cross_entropy_loss(self, images.cuda(),
                                                                Variable(torch.LongTensor(targets_final)).cuda())
                        with torch.no_grad():
                            old_output = self.old_model(images_new)
                        new_output = self.model(images_new)
                        loss_relationship = Loss_relationship(self, old_output, new_output, old_classes)
                        opt.zero_grad()
                        Loss = self.args.loss_weight3 * loss_distribution + self.args.loss_weight1 * loss_cross_entropy + self.args.loss_weight2 * loss_relationship
                        Loss_key = Loss.item()
                        self.lloss = Loss_key

                        with torch.autograd.set_grad_enabled(True):
                            torch.autograd.backward(Loss)

                        opt.step()

                if (self.train_total_data.shape[0] - n) < self.args.batch_size and (
                        self.train_total_data.shape[0] - n) != 0:
                    images, target = self.train_total_data[n:].transpose((0, 3, 1, 2)), self.train_total_label[n:]
                    target = Variable(torch.LongTensor([class_map[label] for label in target]))
                    images = torch.as_tensor(images, dtype=torch.float32).to(self.device)

                    loss_distribution = Loss_distribution(self, images, target, old_classes, new_classes)
                    images, targets_final, _, element, feature_group, images_new, targets_new = pseuable(self, images,
                                                                                                         target,
                                                                                                         old_classes,
                                                                                                         new_classes)
                    loss_cross_entropy = cross_entropy_loss(self, images.cuda(),
                                                            Variable(torch.LongTensor(targets_final)).cuda())
                    with torch.no_grad():
                        old_output = self.old_model(images_new)
                    new_output = self.model(images_new)
                    loss_relationship = Loss_relationship(self, old_output, new_output, old_classes)
                    opt.zero_grad()
                    Loss = self.args.loss_weight3 * loss_distribution + self.args.loss_weight1 * loss_cross_entropy + self.args.loss_weight2 * loss_relationship
                    Loss_key = Loss.item()
                    self.lloss = Loss_key

                    with torch.autograd.set_grad_enabled(True):
                        torch.autograd.backward(Loss)

                    opt.step()
                if epoch % self.args.print_freq == 0:
                    accuracy = self._test()
                    print('epoch:%d, accuracy:%.5f, loss:%.5f' % (epoch, accuracy, self.lloss))
                    # print('epoch:%d, loss:%.5f' % (epoch, self.lloss))

    def _test(self):
        self.model.eval()
        self.test_data, self.test_label = np.array(self.test_data), np.array(self.test_label)
        correct, total = 0.0, 0.0
        n = 0
        for i in range(self.test_data.shape[0]):
            if (i + 1) % self.args.batch_size == 0:
                imgs, labels = self.test_data[n:i + 1].transpose((0, 3, 1, 2)), self.test_label[n:i + 1]
                n += self.args.batch_size
                imgs = torch.as_tensor(torch.from_numpy(imgs), dtype=torch.float32).to(self.device)
                labels = torch.as_tensor(torch.from_numpy(labels), dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    outputs = self.model(imgs)

                predicts = torch.max(outputs, dim=1)[1]
                predicts = torch.tensor([self.map_reverse[pred] for pred in predicts.cpu().numpy()])

                correct += (predicts == labels.cpu()).sum()
                total += len(labels)

        if (self.test_data.shape[0] - n) < self.args.batch_size and (self.test_label.shape[0] - n) != 0:
            imgs, labels = self.test_data[n:].transpose((0, 3, 1, 2)), self.test_label[n:]
            imgs = torch.as_tensor(torch.from_numpy(imgs), dtype=torch.float32).to(self.device)
            labels = torch.as_tensor(torch.from_numpy(labels), dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(imgs)

            predicts = torch.max(outputs, dim=1)[1]
            predicts = torch.tensor([self.map_reverse[pred] for pred in predicts.cpu().numpy()])

            correct += (predicts == labels.cpu()).sum()

            total += len(labels)
        accuracy = correct / total
        self.model.train()
        return accuracy

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        os.makedirs(path, exist_ok=True)
        self.numclass += self.task_size
        filename = f"{path}{self.numclass - self.task_size}_model.pkl"
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

        self.old_new_model = torch.load(filename)

    def random_sample(self, data, label):
        random.seed(2024)
        train_index = list(range(data.shape[0]))
        random.shuffle(train_index)
        random_data, random_label = [], []
        for i in train_index:
            random_data.append(data[i])
            random_label.append(label[i])
        random_data = np.array(random_data)
        random_label = np.array(random_label)
        return random_data, random_label

    def getTrainData(self, classes):
        train_input, train_output = self.train_dataset['train_data'], self.train_dataset['train_labels']
        datas, labels = [], []
        for label in classes:
            for j in range(train_output.shape[1]):
                if train_output[0][j] == label:
                    datas.append(train_input[j])
                    labels.append(label)
        TrainData, TrainLabels = np.array(datas), np.array(labels)
        random_train_data, random_train_label = self.random_sample(TrainData, TrainLabels)
        return random_train_data, random_train_label

    def getTestData(self, classes):
        test_input, test_output = self.test_dataset['train_data'], self.test_dataset['train_labels']
        datas, labels = [], []
        for label in classes:
            for j in range(test_output.shape[1]):
                if test_output[0][j] == label:
                    data = test_input[j]
                    datas.append(data)
                    labels.append(label)
        datas, labels = np.array(datas), np.array(labels)
        self.test_data = datas if self.test_data == [] else np.concatenate((self.test_data, datas), axis=0)
        self.test_label = labels if self.test_label == [] else np.concatenate((self.test_label, labels), axis=0)
        self.test_data, self.test_label = self.random_sample(self.test_data, self.test_label)
