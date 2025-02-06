import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Add pooling layers
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling
        
        # Adjust the input size after pooling (each 24x24 feature map becomes 12x12)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Adjusted fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU after first convolution
        x = self.pool(x)  # Apply pooling
        x = F.relu(self.conv2(x))  # Apply ReLU after second convolution
        x = self.pool(x)  # Apply pooling
      
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor before fully connected layer
        x = F.relu(self.fc1(x))  # Apply ReLU after first fully connected layer
        x = F.relu(self.fc2(x))  # Apply ReLU after second fully connected layer
        x = self.fc3(x)  # Output layer (no activation)
        return x





    
if __name__ == '__main__':
   
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    

    for epoch in range(10):  # Loop through the dataset 2 times (can be increased for more training)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()  # Zero the gradients for this batch
            outputs = net(inputs)  # Forward pass (get predictions)
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update weights

            running_loss += loss.item()
            if i % 2000 == 1999:  # Print the loss every 2000 batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Training Finished!")

    net.eval()  

    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations (faster during evaluation)
        for data in testloader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

    # Visualize some test images and predictions
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # Make predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Plot the images with their predicted and true labels
    # Visualize some test images and predictions
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # Make predictions
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Plot the images with their predicted and true labels
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    for i in range(4):
        ax = axes[i]
        ax.imshow(np.transpose(images[i] / 2 + 0.5, (1, 2, 0)))  # Rescale image back to [0, 1]
        ax.set_title(f'Pred: {predicted[i]} | True: {labels[i]}')
        ax.axis('off')  # Hide axes
    plt.show()
