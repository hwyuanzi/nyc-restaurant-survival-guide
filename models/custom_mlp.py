import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

class CustomMLP(nn.Module):
    """
    A custom Multi-Layer Perceptron implemented from scratch using PyTorch primitives.
    This fulfills the requirement: 'At least one algorithm from the course must be implemented
    without only being a wrapper around a library.'
    We use this model to predict the Health Grade of a restaurant based on its features.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=3, dropout=0.2):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if torch.is_floating_point(x):
            # Keep a single missing feature from poisoning the whole prediction path.
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

def train_mlp(model, X_train, y_train, epochs=50, lr=0.001):
    """
    Training loop for the custom MLP model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        history.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return model, history

def evaluate_mlp(model, X_test, y_test):
    """
    Evaluates the model and computes the F1 score.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        
        f1 = f1_score(y_test.numpy(), predicted.numpy(), average='weighted')
        print(f'Weighted F1 Score: {f1:.4f}')
    return f1, predicted.numpy()


def find_counterfactual(model, base_features, target_class=0, steps=150, lr=0.1):
    """
    Finds the minimum perturbation required (adversarial example) to change
    the prediction of `base_features` to `target_class` (A-Grade).
    We freeze the model weights and optimize the input features directly.
    """
    model.eval()
    
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
        
    # Clone input and enable gradients on it
    opt_features = base_features.clone().detach()
    opt_features.requires_grad = True
    
    optimizer = optim.Adam([opt_features], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    target_tensor = torch.tensor([target_class], dtype=torch.long)
    
    for step in range(steps):
        optimizer.zero_grad()
        output = model(opt_features)
        
        # We want to MINIMIZE the loss between our output and the TARGET class (Grade A)
        loss = criterion(output, target_tensor)
        
        # Add L2 penalty so the counterfactual stays realistically close to the original features
        l2_penalty = 0.5 * torch.sum((opt_features - base_features)**2)
        total_loss = loss + l2_penalty
        
        total_loss.backward()
        optimizer.step()
        
        # Clip continuous features to realistically positive bounds
        with torch.no_grad():
            opt_features.clamp_(min=0.0) 
            
    # Unfreeze model weights for safety
    for param in model.parameters():
        param.requires_grad = True
        
    return opt_features.detach()
