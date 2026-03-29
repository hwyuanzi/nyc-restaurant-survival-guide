import torch
import pytest
from models.custom_mlp import CustomMLP, train_mlp, evaluate_mlp

def test_mlp_forward_shape():
    """
    Test that the MLP produces the correct output tensor dimensions.
    """
    # 10 features, 3 class outputs
    model = CustomMLP(input_dim=10, hidden_dim=32, output_dim=3)
    dummy_input = torch.randn(5, 10) # Batch of 5
    out = model(dummy_input)
    assert out.shape == (5, 3), "Output shape should be (batch_size, output_dim)"

def test_mlp_training_loop():
    """
    Test that the MLP gradients flow and loss decreases over a tiny dataset overfit.
    """
    model = CustomMLP(input_dim=10, hidden_dim=32, output_dim=3)
    
    # Overfit a tiny batch of random dummy data
    torch.manual_seed(42)
    X_train = torch.randn(10, 10)
    y_train = torch.randint(0, 3, (10,))
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate initial loss
    model.eval()
    with torch.no_grad():
        initial_outputs = model(X_train)
        initial_loss = criterion(initial_outputs, y_train).item()
    
    # Train
    model, history = train_mlp(model, X_train, y_train, epochs=20, lr=0.01)
    
    # Evaluate final loss
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_train)
        final_loss = criterion(final_outputs, y_train).item()
    
    assert final_loss < initial_loss, "Model should minimize training loss over time"
