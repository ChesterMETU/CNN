import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pathlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from model import CNN 

#First Model
# conv1 3 32
# conv2 32 64

# fc1 64*12*12 512
# fc2 512 num_classes

# maxpool 2 2
# dropout 0.3

# L2 regularization 1e-4
# Adam optimizer
# Learning rate 0.001

# Transforms: RandomHorizantalFlip, RandomRotation(10), RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2) Normalization(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Dataset has 92 classes

# 30 epochs

# Loss is 0.4532 Accuracy is 82.68% on train set
# Accuracy is 82.14% on test set
#------------------------------------------------------------------
# Second Model
#conv1 3 16
#conv2 16 32
#conv3 32 64

#fc1 64*12*12 512
#fc2 512 256
#fc3 256 num_classes

# maxpool 2 2
# dropout 0.3

# L2 regularization 1e-4
# Adam optimizer
# Learning rate 0.001

# Original image, augmented transform
# Transforms: RandomHorizantalFlip, RandomRotation(10), RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2) Normalization(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Dataset has 92 classes

# 24 epochs

# Loss is 0.3252 Accuracy is 85.47% on train set
# Accuracy is 82.90% on test set
#------------------------------------------------------------------
# Third Model
#conv1 3 16
#conv2 16 32
#conv3 32 64

#fc1 64*12*12 512
#fc2 512 256
#fc3 256 num_classes

# maxpool 2 2
# dropout 0.3

# L2 regularization 1e-4
# Adam optimizer
# Learning rate 0.001

# Original image, augmented transform,already transformed
# Transforms: RandomHorizantalFlip, RandomRotation(10), RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2) Normalization(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Dataset has 92 classes

# 30 epochs

# Loss is 0.2372 Accuracy is 89.36% on train set
# Accuracy is 84.47% on test set
#------------------------------------------------------------------
# Fourth Model
#conv1 3 16
#conv2 16 32
#conv3 32 64

#fc1 64*12*12 512
#fc2 512 256
#fc3 256 num_classes

# maxpool 2 2
# dropout 0.3
# L2 regularization 1e-4
# Adam optimizer
# Learning rate 0.001

# Original image, augmented transform
# Transforms: RandomHorizantalFlip, RandomRotation(10), RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2) Normalization(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Dataset has 92 classes
# 30 epochs
# Loss is 0.2372 Accuracy is 89.36% on train set
# Accuracy is 84.47% on test set
#------------------------------------------------------------------

def evaluate_test_loss(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    
    avg_test_accuracy = 100 * correct / total_samples

    avg_test_loss = total_loss / total_samples
    return avg_test_loss, avg_test_accuracy


torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    np.set_printoptions(precision=3)

    seed = 42
    torch.manual_seed(seed)


    original_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    augmented_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    rotation_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    perspective_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomPerspective(distortion_scale=0.5, p=1, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])




    trainset = datasets.ImageFolder(root='Dataset/Training/', transform=original_transforms)
    trainset_rotation = datasets.ImageFolder(root='Dataset/Training/', transform=rotation_transforms)
    trainset_perspective = datasets.ImageFolder(root='Dataset/Training/', transform=perspective_transforms)
    #augmented_transforms = datasets.ImageFolder(root='Dataset/Training/', transform=augmented_transforms)
    #train_rotation_transforms = datasets.ImageFolder(root='Dataset/Training/', transform=rotation_transforms)
    #train_perspective_transforms = datasets.ImageFolder(root='Dataset/Training/', transform=perspective_transforms)
    #already_transformed = datasets.ImageFolder(root='Dataset/augumented/Training', transform=original_transforms)
    combined_train_data = torch.utils.data.ConcatDataset([trainset, trainset_rotation, trainset_perspective])

    testset = datasets.ImageFolder(root='Dataset/Test/', transform=original_transforms)

    trainloader = torch.utils.data.DataLoader(combined_train_data, batch_size=1024, shuffle=True,pin_memory=True, num_workers=8,multiprocessing_context='forkserver')
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=True,pin_memory=True, num_workers=8,multiprocessing_context='forkserver')

    print("Number of classes(Train Test):", len(trainset.classes), len(testset.classes))
    print("Number of training samples:", len(trainset))
    print("Number of test samples:", len(testset))


    print("----------------------------------------------------------")
    """
    for classname in trainset.classes:
        print(f"Class name: {classname}")
    print("----------------------------------------------------------")
    """
    

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = CNN(num_classes=len(trainset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4 ,lr=0.001)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    correct_train = 0
    train_accuracies = []
    train_loss = []
    test_accuracys = []
    test_losses = []

    print("Training the model...")
    print("----------------------------------------------------------")
    prev_loss = float('inf')
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct_train = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)  # get class with highest score
            correct_train += (predicted == labels).sum().item()

            running_loss += loss.item() * images.size(0) 

        epoch_loss= running_loss / len(trainloader.dataset)  # average loss over dataset
        train_loss.append(epoch_loss)
        scheduler.step(epoch_loss)

        train_acc = correct_train / len(trainloader.dataset)  # calculate accuracy
        train_accuracies.append(train_acc)
        test_loss, test_accuracy = evaluate_test_loss(model, testloader, criterion, device)
        test_losses.append(test_loss)
        test_accuracys.append(test_accuracy)
    
        print(f"Epoch {epoch+1}, Train Accuracy: {train_acc*100}%")

        if (epoch_loss) > prev_loss:
            print(f"Epoch [{epoch + 1}/10], Loss: {epoch_loss}")
            break
        prev_loss = epoch_loss

        print(f"Epoch [{epoch + 1}], Loss: {epoch_loss}")
        print(f"Epoch [{epoch + 1}], Test Loss: {test_loss}")
        print(f"Epoch [{epoch + 1}], Test Accuracy: {test_accuracy}")


    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.plot(epochs, test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    model.eval()  
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Test Accuracy: {accuracy:.2f}%")

    torch.save(model, 'plant_model.pth')
    print("Model saved as plant_model.pth")
