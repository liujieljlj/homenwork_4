import torch 
from torch.autograd import Variable
import torchvision  
from tqdm import tqdm
from utility.util import accuracy

class Tester():
    def __init__(self, test_loader, net, args):
        self.test_loader = test_loader
        self.net = net
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        self.net.to(self.device)

    def test(self):
        epoch_acc = [] # Predict the correct number of images
        self.net.eval() 
        for data in tqdm(self.test_loader, desc="Test Iteration", ncols=70):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.net(Variable(images))
            acc = accuracy(outputs, labels)
            epoch_acc.append(acc.item())
        print('val_acc: %d %%' % (sum(epoch_acc) / len(epoch_acc)))