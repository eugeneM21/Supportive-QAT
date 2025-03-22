import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18

# Step 1: Load the CIFAR-10 dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load training and testing data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def round_up_ste(x: torch.Tensor):
    return (x.ceil() - x).detach() + x

def round_down_ste(x: torch.Tensor):
    return (x.floor() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


def quantize_ste(x: torch.Tensor, num_bits: int, min_val=None, max_val=None):
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()

    # Define scale and zero point for quantization
    qmin, qmax = 0, (2 ** num_bits) - 1
    scale = (max_val - min_val) / (qmax - qmin)
    
    # Normalize and round
    x_q = (x - min_val) / scale  # Normalize to [0, qmax]
    x_q = clamp_ste(round_ste(x_q), qmin, qmax)

    return x_q, min_val, scale

def dequantize_ste(x_q: torch.Tensor, offset: float, scale: float):
    x_deq = x_q * scale + offset
    return x_deq

def fake_quantize(x):
    nx, offset, scale = quantize_ste(x, 2)
    return dequantize_ste(nx, offset, scale)

# Step 2: Define the CNN model (simple architecture for CIFAR-10)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 512)  # Adjusted for 32x32 input (like CIFAR-10)
        self.fc2 = nn.Linear(512, 10)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.batch_norm6 = nn.BatchNorm2d(512)
        self.batch_norm7 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = fake_quantize(x)
        x = self.pool(torch.relu(self.batch_norm2(self.conv2(x))))
        x = fake_quantize(x)
        
        x = torch.relu(self.batch_norm3(self.conv3(x)))
        x = fake_quantize(x)
        
        x = self.pool(torch.relu(self.batch_norm4(self.conv4(x))))
        x = fake_quantize(x)
        
        x = torch.relu(self.batch_norm5(self.conv5(x)))
        x = fake_quantize(x)
        
        x = self.pool(torch.relu(self.batch_norm6(self.conv6(x))))
        x = fake_quantize(x)
        
        x = torch.relu(self.batch_norm7(self.conv7(x)))
        x = fake_quantize(x)

        x = x.view(x.shape[0], -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = fake_quantize(x)
        x = self.fc2(x)
        x = fake_quantize(x)
        return x

# Initialize the model
# model = SimpleCNN().cuda()  # Move model to GPU if available

class EASGD():
    def __init__(self, model_class, num_models):
        super().__init__()
        self.alpha = 0.01
        self.beta = 0.9
        self.num_models = num_models

        self.main_model = model_class().cuda()
        self.main_optimizer = optim.Adam(self.main_model.parameters(), lr=0.001)

        self.models = nn.ModuleList()
        self.optimizers = []

        for _ in range(num_models):
            model = model_class().cuda()
            model.load_state_dict(self.main_model.state_dict())
            self.models.append(model)
            self.optimizers.append(optim.Adam(model.parameters(), lr=0.001))
            
    def _initialize_weights(self):
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.main_model.apply(init_func)
        for model in self.models:
            model.load_state_dict(self.main_model.state_dict())

    def synchronize_models(self):
        with torch.no_grad():
            for model in self.models:
                for local_param, central_param in zip(model.parameters(), self.main_model.parameters()):
                    diff = local_param.data - central_param.data
                    local_param.data -= self.alpha * diff
                    central_param.data += (self.beta / self.num_models) * diff

    def train_easgd(self, trainloader, testloader, num_epochs):
        self._initialize_weights()
        
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            model_choice = 0
            train_loss = 0.0

            for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                model = self.models[model_choice]
                optimizer = self.optimizers[model_choice]

                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                model_choice = (model_choice + 1) % self.num_models

                if model_choice == 0:
                    self.synchronize_models()            
            
            train_loss = train_loss / len(trainloader)
            train_losses.append(train_loss)

            # Step 5: Validation after every epoch
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.main_model(inputs)
                    test_loss += criterion(outputs, labels).item()

            test_loss = test_loss / len(testloader)
            test_losses.append(test_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')


        return train_losses, test_losses
    
    def train_regular(self, trainloader, testloader, num_epochs):
        self._initialize_weights()
        
        criterion = nn.CrossEntropyLoss()
        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            train_loss = 0.0

            for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                model = self.main_model
                optimizer = self.main_optimizer

                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            
            train_loss = train_loss / len(trainloader)
            train_losses.append(train_loss)

            # Step 5: Validation after every epoch
            test_loss = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.main_model(inputs)
                    test_loss += criterion(outputs, labels).item()

            test_loss = test_loss / len(testloader)
            test_losses.append(test_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')


        return train_losses, test_losses

    
num_epochs = 10
easgd = EASGD(resnet18, num_models=2)
train_losses, test_losses = easgd.train_easgd(trainloader, testloader, num_epochs=num_epochs)
train_losses_reg, test_losses_reg = easgd.train_regular(trainloader, testloader, num_epochs=num_epochs)

# Step 6: Plotting Loss and Accuracy
# Plot training loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), train_losses_reg, label='Training Loss Regular')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot test accuracy
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.plot(range(num_epochs), test_losses_reg, label='Test Loss Regular')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("losses.png")