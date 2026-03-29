import torch
import torch.nn as nn
import torch.optim as optim

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

def train_autoencoder(model, X_train, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstruction, _ = model(X_train)
        loss = criterion(reconstruction, X_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'AE Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    return model
