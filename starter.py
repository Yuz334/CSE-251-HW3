from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import time


# TODO: Some missing values are represented by '__'. You need to fill these up.

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 32, num_workers=  0, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size= 32, num_workers= 0, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size= 32, num_workers= 0, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.normal_(m.bias.data)        

epochs = 256
criterion = nn.CrossEntropyLoss()   # Choose an appropriate loss function, we use cross entropy loss
fcn_model = FCN(n_class = n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr = 0.001) # lr is learning rate

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()

        
def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = X.to('cuda:0')
                labels = Y.to('cuda:0')
            else:
                inputs, labels = X, Y
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')
        val(epoch)
        fcn_model.train()
    


def val(epoch):
    use_gpu = torch.cuda.is_available()
    fcn_model.eval()
    score_max = torch.nn.Softmax(dim = 1)
    for iter, (X, tar, Y) in enumerate(val_loader):
        if use_gpu:
            X, tar = X.cuda(), tar.cuda()
        output = score_max(fcn_model(X))
        Ious = iou(output, tar)
        pi_acc = pixel_acc(output, tar)
        print('Validation \n')
        print(Ious)
        print('Validation accuracy' + str(pi_acc))
        break

    #Complete this function - Calculate loss, accuracy and IoU for every epoch

    # Make sure to include a softmax after the output from your model
    
def test():
    fcn_model.eval()
    score_max = torch.nn.Softmax(dim=1)
    for iter, (X, tar, label) in test_loader:
        output = score_max(fcn_model(X))
        Ious = iou(output, tar)
        pi_acc = pixel_acc(output, tar)
        print('Test iou is \n')
        print(Ious)
        print('Test acc is ' + str(pi_acc))



    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    test()