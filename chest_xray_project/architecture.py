import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np


#Data Loader

train_data_path = 'chest_xray/train'
test_data_path = 'chest_xray/test'
batch_size = 32
num_epochs = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose(
    [transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])


train_dataset = ImageFolder(root=train_data_path, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                        shuffle=True, num_workers=0)

testset = ImageFolder(root=test_data_path, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=0)

classes = ('NORMAL', 'PNEUMONIA')

# #view images check if dataloader working
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # Function to show an image
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # Show images
# imshow(torchvision.utils.make_grid(images[0]))
# # Print labels
# print(' '.join(f'{classes[labels[j]]}' for j in range(batch_size)))


#Neural Network Layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 1, stride = 1) 

        self.pool = nn.MaxPool2d(kernel_size = 2) 

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, padding = 1, stride = 1) 

        # Dummy forward pass to calculate the size of the flattened tensor
        dummy_input = torch.autograd.Variable(torch.zeros(1, 1, 224, 224))
        dummy_output = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input))))))
        n_size = dummy_output.data.view(1, -1).size(1)
        
        # Fully connected layers with dynamically calculated input size
        self.fc1 = nn.Linear(n_size, 120)  # Use the dynamically calculated size
        # Other fully connected layers...

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Apply conv1 -> ReLU -> pool

        x = self.pool(F.relu(self.conv2(x))) # Apply conv2 -> ReLU -> pool

        x = torch.flatten(x, 1) # Flatten all dimensions except batch

        x = F.relu(self.fc1(x)) # Apply fc1 -> ReLU


        return x

net = Net()

#optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
total_step = len(trainloader)

#train
def train_and_test():
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    total_step = len(trainloader)

    # Training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    # Testing loop
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the {} train images: {:.2f} %'.format(total, 100 * correct / total))

    # Save the trained model
    torch.save(net.state_dict(), 'model_weights.pth')

if __name__ == "__main__":
    train_and_test()


