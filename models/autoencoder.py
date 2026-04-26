import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class RestaurantAutoencoder(nn.Module):
    """
    Deep Autoencoder to compress high-dimensional restaurant features into a 2D latent space
    for visual exploration (Week 6 concept).
    """
    def __init__(self, input_dim, latent_dim=2):
        super(RestaurantAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent
        
    def get_latent_space(self, x):
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def get_intermediate_embedding(self, x):
        """
        Extract the 32-dimensional intermediate representation from the encoder.
        This is the output after encoder layers [Linear(6→64), ReLU, Linear(64→32)]
        but BEFORE the final compression to latent_dim (2D).
        Useful for PCA analysis on a richer, higher-dimensional learned representation.
        """
        self.eval()
        with torch.no_grad():
            h = self.encoder[0](x)   # Linear(input_dim → 64)
            h = self.encoder[1](h)   # ReLU
            h = self.encoder[2](h)   # Linear(64 → 32)
            return h

def train_autoencoder(model, X_train, epochs=50, lr=0.001, batch_size=256, shuffle=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    effective_batch_size = max(1, min(int(batch_size), len(X_train)))
    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=effective_batch_size,
        shuffle=shuffle,
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_items = 0
        for (batch_X,) in train_loader:
            optimizer.zero_grad()
            reconstruction, _ = model(batch_X)
            loss = criterion(reconstruction, batch_X)
            loss.backward()
            optimizer.step()

            batch_items = int(batch_X.shape[0])
            epoch_loss += float(loss.item()) * batch_items
            total_items += batch_items

        if (epoch + 1) % 10 == 0:
            mean_loss = epoch_loss / max(total_items, 1)
            print(f'AE Epoch [{epoch+1}/{epochs}], Loss: {mean_loss:.4f}')
            
    return model
