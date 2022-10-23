from train import Trainer
from test import Tester 
from model import LeNet, VGG, ResNet
from data.dataset import DataSet, DataBuilder 
from utility.util import check_path, show_model

import torch
import torch.nn as nn 
from torch import optim
import argparse
import os


def main(args):

    check_path(args)

    # All categories of CIFAR-10, 10 categories
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # # Preparing the Data
    data_builder = DataBuilder(args)
    datasets = DataSet(data_builder.train_builder(), data_builder.test_builder(), classes)
    
    # Building Neural Networks
    if args.lenet:
        net = LeNet.LeNet()
        model_name = args.name_lenet
    if args.vgg:
        net = VGG.Vgg16_Net()
        model_name = args.name_vgg
    if args.resnet:
        net = ResNet.ResNet9(in_channels=3, num_classes=10)
        model_name = args.name_resnet

    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # SGD
    optimizer = optim.SGD(
        net.parameters(), 
        lr=args.learning_rate, 
        momentum=args.sgd_momentum, 
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, epochs=args.epoch, 
                                                steps_per_epoch=len(datasets.train_loader))
    # Model Saving Path
    model_path = os.path.join(args.model_path, model_name)

    # GPU / CPU
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    # Training
    if args.do_train:
        print("Training...")
        trainer = Trainer(net, criterion, optimizer, scheduler, datasets.train_loader, args)
        trainer.train(epochs=args.epoch)
        torch.save(net.state_dict(), model_path)
    
    # Testing
    if args.do_eval:
        if not os.path.exists(model_path):
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        print("Testing...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load(model_path, map_location=device))
        # net.eval()
        tester = Tester(datasets.test_loader, net, args)
        tester.test()
    
    if args.show_model:
        if not os.path.exists(model_path):
            print("Sorry, there's no saved model yet, you need to train first.")
            return
        show_model(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   
    # Dataset
    parser.add_argument("--num_workers", default=0, type=int, help="Thread number for training.")
    parser.add_argument("--is_download", default=True, type=bool, help="Download the datasets if there is no data.")

    # path
    parser.add_argument("--data_path", default="cifar", type=str, help="The directory of the CIFAR-10 data.")
    parser.add_argument("--model_path", default="ckpt", type=str, help="The directory of the saved model.")
    
    # model
    parser.add_argument("--name_lenet", default="state_dict_le", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_vgg", default="state_dict_vgg", type=str, help="The name of the saved model's parameters.")
    parser.add_argument("--name_resnet", default="state_dict_resnet", type=str, help="The name of the saved model's parameters.")

    # Training Relation
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--epoch", default=10, type=int, help="The number of training epochs.")
    parser.add_argument("--seed", default=42, type=int, help="The random seed used for initialization.")

    #### epoch, iteration, batchsize
    
    # params
    parser.add_argument("--learning_rate", default=0.01, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-4, type=int, help="Weight decay of SGD optimzer.")
    parser.add_argument("--sgd_momentum", default=0.9, type=float, help="The momentum of the SGD optimizer.")
    
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="The Epsilon of Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Command
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")
    parser.add_argument("--show_model", action="store_true", help="Display the state dict of the model.")
    parser.add_argument("--lenet", action="store_true", help="Use LeNet-5 as the model.")
    parser.add_argument("--vgg", action="store_true", help="Use VGG-16 as the model.")
    parser.add_argument("--resnet", action="store_true", help="Use ResNet as the model.")

    args = parser.parse_args()
    main(args)