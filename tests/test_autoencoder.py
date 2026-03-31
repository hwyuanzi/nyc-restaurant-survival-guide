import torch
import pytest
from models.autoencoder import RestaurantAutoencoder, train_autoencoder


def test_autoencoder_forward_shape():
    """Test that the autoencoder produces correct output dimensions."""
    model = RestaurantAutoencoder(input_dim=6, latent_dim=2)
    dummy = torch.randn(10, 6)
    reconstruction, latent = model(dummy)
    assert reconstruction.shape == (10, 6), "Reconstruction should match input shape"
    assert latent.shape == (10, 2), "Latent space should be (batch, latent_dim)"


def test_autoencoder_training_reduces_loss():
    """Test that the autoencoder actually learns — reconstruction loss must decrease."""
    torch.manual_seed(42)
    model = RestaurantAutoencoder(input_dim=6, latent_dim=2)
    X = torch.randn(50, 6)

    criterion = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        recon_before, _ = model(X)
        loss_before = criterion(recon_before, X).item()

    model = train_autoencoder(model, X, epochs=30, lr=0.005)

    model.eval()
    with torch.no_grad():
        recon_after, _ = model(X)
        loss_after = criterion(recon_after, X).item()

    assert loss_after < loss_before, "AE reconstruction loss should decrease after training"


def test_get_latent_space():
    """Test that get_latent_space returns correct dimensions without gradients."""
    model = RestaurantAutoencoder(input_dim=6, latent_dim=2)
    X = torch.randn(20, 6)
    latent = model.get_latent_space(X)
    assert latent.shape == (20, 2)
    assert not latent.requires_grad, "Latent output should be detached"


def test_get_intermediate_embedding():
    """Test that the 32-D intermediate embedding extraction works for PCA analysis."""
    model = RestaurantAutoencoder(input_dim=6, latent_dim=2)
    X = torch.randn(20, 6)
    intermediate = model.get_intermediate_embedding(X)
    assert intermediate.shape == (20, 32), "Intermediate embedding should be (batch, 32)"
    assert not intermediate.requires_grad, "Intermediate output should be detached"
