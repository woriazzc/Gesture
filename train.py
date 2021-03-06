from utils import *
from model import Model
import torch.optim as optim
from thop import profile, clever_format
from data_utils import *
import os
import numpy as np

feature_dim = 128
temperature = 0.5
batch_size = 64
epochs = 5
temperature = 0.5
k = 200
C = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './model/model.pt'
bank_path = './model/bank.pt'
label_path = './model/label.pt'


if __name__ == "__main__":
    train_trans = compute_train_transform()
    test_trans = compute_test_transform()

    train_data = GesDataset('datasets/train/', transform=train_trans)
    test_data = GesDataset('datasets/test/', transform=test_trans)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    memory_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # img, lab = train_data[1]
    # print(img.shape)
    # imshow(img, lab)
    # plt.show()

    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('results'):
        os.mkdir('results')

    model = Model(feature_dim)
    try:
        model = torch.load(model_path)
    except:
        pass
    model = model.to(device)

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@2': []}
    best_acc = 0.0
    if os.path.exists('./results/best_acc.npy'):
        best_acc = np.load('./results/best_acc.npy')

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader, epoch, epochs, temperature, batch_size, device)
        test_acc1, test_acc2, feature_bank, feature_label = test(model, memory_dataloader, test_loader, k, epoch, epochs, C, temperature, device)
        results['train_loss'].append(train_loss)
        results['test_acc@1'].append(test_acc1)
        results['test_acc@2'].append(test_acc2)
        if test_acc1 > best_acc:
            best_acc = test_acc1
            torch.save(model, model_path)
            torch.save(feature_bank, bank_path)
            torch.save(feature_label, label_path)

    for k in results:
        try:
            r = np.load('./results/{}.npy'.format(k))
            results[k] = np.append(r, results[k])
        except:
            pass
        np.save('./results/{}.npy'.format(k), results[k])

    np.save('./results/best_acc.npy', best_acc)