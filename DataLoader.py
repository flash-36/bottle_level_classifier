import os
import numpy as np
from PIL import Image


class DataLoader():
    def __init__(self, PATH, size = (128,128)):
        assert os.path.exists(PATH)
        self.train_path = os.path.join(PATH, 'kaggle_train_128/train_128')
        self.test_path = os.path.join(PATH, 'kaggle_test_128/test_128')
        self.size = size

    def load(self, preprocess=False):
        X_train = []
        y_train = []
        X_test = []

        for image_file in os.listdir(self.test_path):
            PILimage = Image.open(os.path.join(self.test_path, image_file))
            PILimage.thumbnail(self.size)
            X_test.append(np.array(PILimage))

        for class_name in os.listdir(self.train_path):
            for image_file in os.listdir(os.path.join(self.train_path, class_name)):
                PILimage = Image.open(os.path.join(self.train_path, class_name, image_file))
                PILimage.thumbnail(self.size)
                image = np.array(PILimage)
                X_train.append(image / 256 - 0.5)
                y_train.append(int(class_name))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        if preprocess:
            X_train = X_train[:, :, 33:93, :]
            X_test = X_test[:, :, 33:93, :]
        X_train = X_train.transpose((0, 3, 1, 2))
        X_test = X_test.transpose((0, 3, 1, 2))
        return X_train, y_train, X_test


if __name__ == '__main__':
    dataloader = DataLoader(
        os.path.join("C:/Users/ujwal/Studies/NN_DL Research/bottle-challenge-e6040-2020/modified_kaggle_HW2"))
    X_train, y_train, X_test = dataloader.load(True)
    print(X_train.shape, y_train.shape, X_test.shape)
    import matplotlib.pyplot as plt

    num = int(input())
    while num:
        plt.imshow(X_train[num - 1])
        plt.title(str(y_train[num - 1]))
        plt.show()
        num = int(input())
