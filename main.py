
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from PIL import Image
from model import CNN  # Assuming you have a model class defined in plant_model.py

transform = transforms.Compose([
    transforms.Resize((100, 100)),  # must match training size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('mps')

model = torch.load('plant_model.pth', weights_only=False)
#model.eval()
model.to(device)
model.eval()

testset = ImageFolder('Dataset/Test/', transform=transform)
#test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#criterion = CrossEntropyLoss()


"""
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():  # disable gradient computation for speed
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        test_loss += loss.item() * images.size(0) # sum loss * batch size
        total += images.size(0)
        print(images.size(0),loss)

        # Accuracy (optional)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        

# Average loss over dataset
avg_test_loss = test_loss / total
# Accuracy
test_accuracy = 100 * correct / total

print(f"📉 Test Loss: {avg_test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.2f}%")
"""
img = Image.open('prediction/busefer.jpg').convert('RGB')  # ensure 3 channels
input_tensor = transform(img).unsqueeze(0)  # add batch dimension

with torch.no_grad():  
    if torch.backends.mps.is_available():  
        mps_device = torch.device("mps:0") 
        input_tensor = input_tensor.to(mps_device) 
        output = model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
'''
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
'''
print(testset.classes)
class_names = testset.classes # e.g. ['apple', 'banana', ...]
predicted_label = class_names[predicted_idx.item()]
print(f"Predicted label: {predicted_label}")

