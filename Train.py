import os
import csv
import torch
import torch.optim as optim
import torch.nn as nn
from Model import Net
from DataLoader import DataLoader
from sklearn.model_selection import train_test_split

PATH = os.path.join("./modified_kaggle_HW2")

net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
net = nn.DataParallel(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
data_gen = DataLoader(PATH)
X_train_load, y_train_load, X_test = data_gen.load(preprocess=True)


def batch_feeder(X_train, y_train, batch_size):
    X_mini = []
    y_mini = []
    for i in range(X_train.shape[0] // batch_size):
        X_mini.append(X_train[i:i + batch_size])
        y_mini.append(y_train[i:i + batch_size])
    print(len(X_mini))
    return zip(X_mini, y_mini)


# Hyper params
batch_size = 100
num_epochs = 10

X_train, X_val, y_train, y_val = train_test_split(X_train_load, y_train_load, test_size=0.3)
for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    i = 0
    for data in batch_feeder(X_train, y_train, batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.from_numpy(inputs).float().to(device)
        labels = torch.from_numpy(labels).long().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        # print("ip",inputs.shape,"op",outputs.shape,"gt",labels.shape)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        i += 1
    print(i)

print('Finished Training')

SAVE_PATH = './prototype_net'
torch.save(net.state_dict(), SAVE_PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in batch_feeder(X_val, y_val, batch_size):
        images, labels = data
        images = torch.from_numpy(images).float().to(device)
        labels = torch.from_numpy(labels).long().to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on validation images: %d %%' % (
        100 * correct / total))

test_images = torch.from_numpy(X_test).float()
outputs = net(test_images)
_, y_pred = torch.max(outputs.data, 1)


def generate_csv(y_pred):
    with open('predicted.csv', 'w') as csvfile:
        fieldnames = ['Id', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, l in enumerate(y_pred):
            filename = str(index).zfill(5) + '.png'
            label = str(l)
            writer.writerow({'Id': filename, 'label': label})


generate_csv(y_pred.cpu().numpy())
