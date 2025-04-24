# import torch
# import torch.nn as nn

# class HandwritingGenerator(nn.Module):
#     def __init__(self):
#         super(HandwritingGenerator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 32, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * 128 * 512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 128 * 512),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.model(x)

import torch.nn as nn

class HandwritingGenerator(nn.Module):
    def __init__(self):
        super(HandwritingGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))  # Smart downsampling
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 8 * 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, 64, 8, 8)  # Reshape for decoding
        x = self.decoder(x)
        return x
