import argparse
import datetime
import random

from Test import main as Test
from Train import  main as Train


def main():
    parser = argparse.ArgumentParser(description='DRDC model')
    parser.add_argument('--epochs1', default=71, type=int, help='Total number of epochs to run')
    parser.add_argument('--epochs2', default=91, type=int, help='Total number of epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
    parser.add_argument('--data_name', default='Dateset name', type=str, help='Dataset name to use')
    parser.add_argument('--total_nc', default=15, type=int, help='class number for the dataset')
    parser.add_argument('--fg_nc', default=8, type=int, help='the number of classes in first task')
    parser.add_argument('--task_num', default=1, type=int, help='the number of incremental steps')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--loss_weight1', default=0.4, type=float, help='loss1 cross_entroy')
    parser.add_argument('--loss_weight2', default=0.4, type=float, help='loss2 relationship')
    parser.add_argument('--loss_weight3', default=0.2, type=float, help='loss3 distribution')
    parser.add_argument('--temperature', default=0.1, type=float, help='training time temperature')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
    parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')

    args = parser.parse_args()
    print(args)
    random.seed(datetime.datetime.now().year)
    for i in range(1):
        print('=======================第', str(i + 1), '次训练模型================')
        Train(args)
        Test(args)


if __name__ == "__main__":
    main()
