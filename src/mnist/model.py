import torch
import torch.nn as nn
import torch.nn.functional as F


class MVAE(nn.Module):
    def __init__(self, z):
        super(MVAE, self).__init__()

        self.image_encoder = ImageEncoder(z)
        self.image_decoder = ImageDecoder(z)
        self.label_encoder = LabelEncoder(z)
        self.label_decoder = LabelDecoder(z)
        self.z_dim = z
        self.poe = ProductOfExperts()

    def forward(self, image_modal=None, label_modal=None):
        # TBD
        if image_modal is not None:
            mu, logvar = self.image_encoder(image_modal)
        if label_modal is not None:
            mu, logvar = self.label_encoder(label_modal)
        # TODO: what is the most elegant way to treat the mu and logvar here
        # TODO: How to realize the product of experts functionality

        print("MU: " + str(mu))
        print("LOGVAR: " + str(logvar))


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
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x):
        h = self.fc1(x.view(-1, 784))
        h = F.relu(h)

        logits = self.fc2(h)
        logits = F.relu(logits)

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
        embed = embed.mean(dim=0)
        # embed = F.relu(embed)

        h = self.fc2(embed)
        h = F.relu(h)

        logits = self.fc3(h)
        logits = F.relu(h)

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

class ProductOfExperts:
    pass

if __name__ == "__main__":
    print("blub")
    enc = ImageEncoder(20)
    image = torch.rand(28, 28)

    mu ,logvar = enc.forward(image)

    print(mu.squeeze(0).size())
    print(mu)
    print(logvar.squeeze(0).size())
    print(logvar)