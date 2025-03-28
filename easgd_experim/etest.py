import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.models import resnet18
from quantize.quantizer import UniformAffineQuantizer, RoundUpQuantizer, RoundDownQuantizer

# Step 1: Load the CIFAR-10 dataset and apply transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def quantize_ste(x: torch.Tensor, num_bits: int, min_val=None, max_val=None):
    if min_val is None:
        min_val = x.min()
    if max_val is None:
        max_val = x.max()
    qmin, qmax = 0, (2 ** num_bits) - 1
    scale = (max_val - min_val) / (qmax - qmin)
    x_q = (x - min_val) / scale
    x_q = clamp_ste(round_ste(x_q), qmin, qmax)
    return x_q, min_val, scale

def dequantize_ste(x_q: torch.Tensor, offset: float, scale: float):
    return x_q * scale + offset

def fake_quantize(x):
    nx, offset, scale = quantize_ste(x, 2)
    return dequantize_ste(nx, offset, scale)

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, quantizer_cls=UniformAffineQuantizer):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.quantizer = quantizer_cls(weight=self.linear.weight, group_size=self.linear.weight.shape[-1])

    def forward(self, x):
        quantized_weight = self.quantizer(self.linear.weight)
        return nn.functional.linear(x, quantized_weight, self.linear.bias)

def replace_linear_layers(model, quantizer_cls):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None
            setattr(model, name, QuantLinear(in_features, out_features, bias, quantizer_cls))
        else:
            replace_linear_layers(module, quantizer_cls)
    return model

class EASGD():
    def __init__(self, model_class, num_models, quantizer_classes=None):
        super().__init__()
        self.alpha = 0.01
        self.beta = 0.9
        self.num_models = num_models

        if quantizer_classes is None:
            quantizer_classes = [UniformAffineQuantizer] * num_models

        self.main_model = replace_linear_layers(model_class(), quantizer_classes[0]).cuda()
        self.main_optimizer = optim.Adam(self.main_model.parameters(), lr=0.001)

        self.models = nn.ModuleList()
        self.optimizers = []

        for i in range(num_models):
            model = replace_linear_layers(model_class(), quantizer_classes[i]).cuda()
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
        train_losses, test_losses = [], []

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

            train_losses.append(train_loss / len(trainloader))
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = self.main_model(inputs)
                    test_loss += criterion(outputs, labels).item()
            test_losses.append(test_loss / len(testloader))

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.2f}, Test Loss: {test_losses[-1]:.2f}")

        return train_losses, test_losses

    def train_regular(self, trainloader, testloader, num_epochs):
        self._initialize_weights()
        criterion = nn.CrossEntropyLoss()
        train_losses, test_losses = [], []

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

            train_losses.append(train_loss / len(trainloader))
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    test_loss += criterion(outputs, labels).item()
            test_losses.append(test_loss / len(testloader))

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.2f}, Test Loss: {test_losses[-1]:.2f}")

        return train_losses, test_losses

num_epochs = 10
quantizer_classes = [RoundUpQuantizer, RoundDownQuantizer]
easgd = EASGD(resnet18, num_models=2, quantizer_classes=quantizer_classes)
train_losses, test_losses = easgd.train_easgd(trainloader, testloader, num_epochs=num_epochs)
train_losses_reg, test_losses_reg = easgd.train_regular(trainloader, testloader, num_epochs=num_epochs)

plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Training Loss (EASGD)')
plt.plot(range(num_epochs), train_losses_reg, label='Training Loss (Regular)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot test loss
plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), test_losses, label='Test Loss (EASGD)')
plt.plot(range(num_epochs), test_losses_reg, label='Test Loss (Regular)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

plt.tight_layout()
plt.savefig("losses.png")
plt.show()