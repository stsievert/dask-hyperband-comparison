import skorch.utils
from skorch import NeuralNetRegressor
import torch.nn as nn
import torch
import skorch


def _initialize(method, layer, gain=1):
    weight = layer.weight.data
    # _before = weight.data.clone()
    kwargs = {'gain': gain} if 'xavier' in str(method) else {}
    method(weight.data, **kwargs)
    # assert torch.all(weight.data != _before)


class Autoencoder(nn.Module):
    def __init__(self, activation='ReLU', init='xavier_uniform_',
                 **kwargs):
        super().__init__()

        self.activation = activation
        self.init = init
        self._iters = 0

        init_method = getattr(torch.nn.init, init)
        act_layer = getattr(nn, activation)
        act_kwargs = {'inplace': True} if self.activation != 'PReLU' else {}

        gain = 1
        if self.activation in ['LeakyReLU', 'ReLU']:
            name = 'leaky_relu' if self.activation == 'LeakyReLU' else 'relu'
            gain = torch.nn.init.calculate_gain(name)

        inter_dim = 28 * 28 // 4
        latent_dim = inter_dim // 4
        layers = [
            nn.Linear(28 * 28, inter_dim),
            act_layer(**act_kwargs),
            nn.Linear(inter_dim, latent_dim),
            act_layer(**act_kwargs)
        ]
        for layer in layers:
            if hasattr(layer, 'weight') and layer.weight.data.dim() > 1:
                _initialize(init_method, layer, gain=gain)
        self.encoder = nn.Sequential(*layers)
        layers = [
            nn.Linear(latent_dim, inter_dim),
            act_layer(**act_kwargs),
            nn.Linear(inter_dim, 28 * 28),
            nn.Sigmoid()
        ]
        for layer in layers:
            if hasattr(layer, 'weight') and layer.weight.data.dim() > 1:
                _initialize(init_method, layer, gain=gain)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        self._iters += 1
        shape = x.size()
        x = x.view(x.shape[0], -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(shape)
    
class NegLossScore(NeuralNetRegressor):
    steps = 0
    def partial_fit(self, *args, **kwargs):
        
        super().partial_fit(*args, **kwargs)
        self.steps += 1
        
    def score(self, X, y):
        X = skorch.utils.to_tensor(X, device=self.device)
        y = skorch.utils.to_tensor(y, device=self.device)
        
        self.initialize_criterion()
        y_hat = self.predict(X)
        y_hat = skorch.utils.to_tensor(y_hat, device=self.device)
        loss = super().get_loss(y_hat, y, X=X, training=False).item()
        print(f'steps = {self.steps}, loss = {loss}')
        return -1 * loss
    
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.callbacks_ = []
        
