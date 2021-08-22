import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    loss = np.load('./results/train_loss.npy')
    acc = np.load('./results/test_acc@1.npy')
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(loss)
    axs[0].set(xlabel='epoch (s)', ylabel='loss', title='train loss')
    axs[1].plot(acc, color='orange')
    axs[1].set(xlabel='epoch (s)', ylabel='accuracy (%)', title='test accuracy')
    plt.show()
    fig.savefig('./results/result.png')