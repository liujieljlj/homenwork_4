import torch 
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, net, criterion, optimizer, scheduler, train_loader, args):
        self.net = net 
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.net.to(self.device)

    def train(self, epochs):
        for epoch in range(epochs):
            print("\n******** Epoch %d / %d ********\n" % (epoch + 1, epochs))
            running_loss = 0.0
            iteration = tqdm(self.train_loader, desc="Train Iteration", ncols=70)
            for i, data in enumerate(iteration):

                # input data
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                # forward 
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels).to(self.device)
                # backward
                loss.backward()

                # updata parameters
                self.optimizer.step()
                # print the information of training
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        self.scheduler.step()
        print('\nFinish training\n') 