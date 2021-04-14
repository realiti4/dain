import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Deep_DAIN_Layer(nn.Module):
    def __init__(self, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=8, layer_dim=1):
        super(Deep_DAIN_Layer, self).__init__()
        print(f'Dain: Mode = {mode}, Layer Dim = {layer_dim}')

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average        
        self.mean_layer = nn.Sequential(*[nn.Linear(input_dim, input_dim, bias=False) for i in range(layer_dim)])
        self._init_weights(self.mean_layer)

        # Parameters for adaptive std
        self.scaling_layer = nn.Sequential(*[nn.Linear(input_dim, input_dim, bias=False) for i in range(layer_dim)])
        self._init_weights(self.scaling_layer)

        # Parameters for adaptive scaling
        self.gating_layer = nn.Sequential(*[nn.Linear(input_dim, input_dim) for i in range(layer_dim)])

        self.eps = 1e-8

    def _init_weights(self, layer):
        for i in range(len(layer)):
            torch.nn.init.eye_(layer[i].weight)

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.view(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.view(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            # assert adaptive_avg.dtype != torch.float16, "don't use fp16 for dain"
            adaptive_avg = adaptive_avg.view(adaptive_avg.size(0), adaptive_avg.size(1), 1)            
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x.pow(2), 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.view(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:            
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.view(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x.pow(2), 2)
            std = torch.sqrt(std + self.eps)
            # std = x.std(2)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.view(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std

            # Step 3: 
            avg = torch.mean(x, 2)
            gate = torch.sigmoid(self.gating_layer(avg))
            gate = gate.view(gate.size(0), gate.size(1), 1)

            x = x * gate

        else:
            assert False

        return x