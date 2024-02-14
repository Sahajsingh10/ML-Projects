import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import architecture as arch

val_data_path = 'chest_xray/val'

transform = transforms.Compose(
    [transforms.Resize((224, 224)), 
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

testval = ImageFolder(root=val_data_path, transform=transform)

val_loader = torch.utils.data.DataLoader(testval, batch_size=8,
                                        shuffle=True, num_workers=0)

net = arch.Net()
net.load_state_dict(torch.load('model_weights.pth'))

# # Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of test data
dataiter = iter(val_loader)
images, labels = next(dataiter)



# Choose the first image in the batch
image, true_label = images[6], labels[6]

# Display the image
imshow(torchvision.utils.make_grid(image))

# Move the image and model to the same device
image = image.to(arch.device).unsqueeze(0)  # Add a batch dimension
net.to(arch.device)
net.eval()  # Set the model to evaluation mode

# Get the model's prediction
with torch.no_grad():
    outputs = net(image)
    _, predicted = torch.max(outputs, 1)

# Map the predicted and true labels to their class names if necessary
predicted_label = arch.classes[predicted.item()]
true_label_name = arch.classes[true_label.item()]

# Print the results
print(f'Predicted: "{predicted_label}", True: "{true_label_name}"')