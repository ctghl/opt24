import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import build_model, classes

batch_size = 2
num_epochs = 5  
model_path = 'models/cifar_model.pth'

def show_images(trainloader):
    def imshow(img):
        img = img / 2 + 0.5  
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

def train(optimizer_type='SGD'):
    print(f'Load data for {optimizer_type}')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    net = build_model()
    criterion = nn.CrossEntropyLoss()

    # Выбор оптимизатора
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    elif optimizer_type == 'Adamax':
        optimizer = optim.Adamax(net.parameters(), lr=0.002)

    losses = []  

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        average_loss = running_loss / len(trainloader)
        losses.append(average_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    print(f'Finished Training with {optimizer_type}')
    return losses 

if __name__ == '__main__':
    sgd_losses = train(optimizer_type='SGD')
    adamax_losses = train(optimizer_type='Adamax')

    # Построение графика потерь
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), sgd_losses, label='SGD', marker='o')
    plt.plot(range(1, num_epochs + 1), adamax_losses, label='Adamax', marker='o')

    plt.title('Loss vs. Epochs for SGD and Adamax Optimizers')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_vs_epochs.png') 
    plt.show()  

    torch.save({'sgd': sgd_losses, 'adamax': adamax_losses}, 'losses.pth')
