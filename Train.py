import os
import torch
import torch.optim as optim
import torch.nn as nn
from Model import Net
from DataLoader import DataLoader
from sklearn.model_selection import train_test_split

PATH = os.path.join("C:/Users/ujwal/Studies/NN_DL Research/bottle-challenge-e6040-2020/modified_kaggle_HW2")

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
data_gen = DataLoader(PATH)
X_train_load, y_train_load, X_test = data_gen.load()


def batch_feeder(X_train, y_train, batch_size):
    X_mini = []
    y_mini = []
    for i in range(X_train.shape[0] // batch_size):
        X_mini.append(X_train[i:i + batch_size])
        y_mini.append(y_train[i:i + batch_size])
    return zip(X_mini, y_mini)


# Hyper params
batch_size = 4
num_epochs = 2

X_train, X_val, y_train, y_val = train_test_split(X_train_load, y_train_load, test_size=0.3)
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    i = 0
    for data in batch_feeder(X_train, y_train, batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.from_numpy(inputs).float()
        labels = torch.from_numpy(labels).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        i+=1

print('Finished Training')

SAVE_PATH = './cifar_net.pth'
torch.save(net.state_dict(), SAVE_PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in batch_feeder(X_val, y_val, batch_size):
        images, labels = data
        images = torch.from_numpy(images).float()
        labels = torch.from_numpy(labels).long()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))