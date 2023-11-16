import torch
import torch.nn as nn


class MVAE(nn.Module):
    def __init__(self, z):
        super(MVAE, self).__init__()

        self.image_encoder = ImageEncoder(z)
        self.image_decoder = ImageDecoder(z)
        self.label_encoder = LabelEncoder(z)
        self.label_decoder = LabelDecoder(z)
        self.poe = ProductOfExperts()

    def forward(self, image_modal, label_modal):
        # TBD
        mu, logvar = self.smth 
        # TODO: what is the most elegant way to treat the mu and logvar here
        # TODO: How to realize the product of experts functionality

        z = self.reparameterize(mu, logvar)
        gen_image = self.image_decoder(z)
        gen_label = self.label_decoder(z)

        return gen_image, gen_label, mu, logvar

    def reparameterize(self):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return eps * std + mu


    
class ImageEncoder(nn.Module):
    def __init__(self, z_dim):
        super(ImageEncoder, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512*2, n_latent*2)

    def forward(self, x):
        h = self.fc1(x.view(-1, 784))
        h = nn.ReLU(h)

        logits = self.fc2(h)
        logits = nn.ReLU(logits)

        mu = self.fc31(logits)
        var = self.fc32(logits)

        return mu, var # This is probably wrong, why is it this way in the picture though
        # NOTE: seems as this is the correct way (theoretically). fc31 is basically mu and fc32 is var

class ImageDecoder(nn.Module):
    def __init__(self, z_dim):
        super(ImageDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 784)

    def forward(self, x):
        h = self.fc1(x)
        h = nn.ReLU(h)

        h = self.fc2(h)
        h = nn.ReLU(h)

        logits = self.fc3(h)
        logits = nn.ReLU(logits)
        # sigmoid probably computing during training

        return logits

class LabelEncoder(nn.Module):
    def __init__(self, z_dim):
        super(LabelEncoder, self).__init__()
        self.fc1 = nn.Embedding(10, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc41 = nn.Linear(512, z_dim)
        self.fc42 = nn.Linear(512, z_dim)

    def forward(self, x):
        embed = self.fc1(x)
        h = self.fc2(embed)
        h = nn.ReLU(h)

        logits = self.fc3(h)
        logits = nn.ReLU(h)

        mu = self.fc41(logits)
        var = self.fc42(logits)

        return mu, var

class LabelDecoder(nn.Module):
    def __init__(self, z_dim):
        super(LabelDecoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        h = self.fc1(x)
        h = nn.ReLU(h)

        h = self.fc2(h)
        h = nn.ReLU(h)

        logits = self.fc3(h)
        logits = nn.ReLU(logits)
        # softmax probably better to use during training function

        return logits