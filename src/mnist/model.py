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

    def forward(self, image_modal=None, label_modal=None):
        # TBD
        mu, logvar = self.prepare_poe(image_modal, label_modal)
        print("LOGVAR")
        print(logvar.size())
        print("MU")
        print(mu.size())
        poe_mu, poe_logvar = self.compute_poe(mu, logvar)
    
        z = self.reparameterize(poe_mu, poe_logvar)
        print("Z: ")
        print(z.size())
        print(z)
        
        if image_modal is not None:
            gen_image = self.image_decoder(z)
        else:
            gen_image = None
        if label_modal is not None:
            gen_label = self.label_decoder(z)
        else:
            gen_label = None
        
        print("GEN IMAGE")
        print(gen_image.size())

        return gen_image, gen_label, mu, logvar

    def reparameterize(self, poe_mu, poe_logvar):
        print("#"*20)
        print(poe_logvar.size())
        std = torch.exp(0.5*poe_logvar)
        print(std.size())
        eps = torch.randn_like(std)
        print("EPS: ")
        print(eps.size())
        return eps * std + poe_mu
    
    def prior_experts(self, n_samples):
        normal_mu = torch.zeros((n_samples, self.z_dim))
        normal_logvar = torch.zeros((n_samples, self.z_dim))
        return normal_mu, normal_logvar

    # The heart of the implementation
    def compute_poe(self, mu, logvar, eps=1e-8):
        # VAE learns logvar, but we need var
        var = torch.exp(logvar) + eps
        print(var.size())
        # In paper T is inverse of covariance matrix, which is 1/var
        T = 1. / var

        cov_term = torch.sum(T, dim=0)
        print("denom: " + str(cov_term.size()))
        poe_cov = 1 / cov_term
        poe_mu = poe_cov * torch.sum(mu * T, dim=0)

        # Get back the logvar to align with learning objective
        poe_logvar = torch.log(poe_cov)

        return poe_mu, poe_logvar


    def prepare_poe(self, image_modal, label_modal):
        if image_modal is None and label_modal is None:
            assert("Both modalities are None. At least one modality must be present.")
        
        mus, logvars = [], []
        if image_modal is not None:
            img_mu, img_logvar = self.image_encoder(image_modal)
            mus.extend([img_mu])
            logvars.extend([img_logvar])
            n_samples = img_mu.size(0)

        if label_modal is not None:
            label_mu, label_logvar = self.label_encoder(label_modal)
            mus.extend([label_mu])
            logvars.extend([label_logvar])
            n_samples = label_mu.size(0)
        
        prior_mu, prior_logvar = self.prior_experts(n_samples)
        print("PRIOR MU")
        print(prior_mu.size())
        print("PRIOR LOGVAR")
        print(prior_logvar.size())
        mus.extend([prior_mu]), logvars.extend([prior_logvar])

        stacked_mu = torch.stack(mus)
        print("STACKED MU")
        print(stacked_mu.size())
        stacked_logvars = torch.stack(logvars)

        return stacked_mu, stacked_logvars




    
class ImageEncoder(nn.Module):
    def __init__(self, z_dim):
        super(ImageEncoder, self).__init__()
        self.l_input = nn.Linear(784, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.out_mu = nn.Linear(512, z_dim)
        self.out_logvar = nn.Linear(512, z_dim)

    def forward(self, x):
        x = self.l_input(x.view(-1, 784))
        x = F.relu(x)

        x = self.l_hidden(x)
        x = F.relu(x)

        logits_mu = self.out_mu(x)
        logits_logvar = self.out_logvar(x)

        return logits_mu, logits_logvar # This is probably wrong, why is it this way in the picture though
        # NOTE: seems as this is the correct way (theoretically). fc31 is basically mu and fc32 is var

class ImageDecoder(nn.Module):
    def __init__(self, z_dim):
        super(ImageDecoder, self).__init__()
        self.l_input = nn.Linear(z_dim, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.l_output = nn.Linear(512, 784)

    def forward(self, x):
        x = self.l_input(x)
        x = F.relu(x)

        x = self.l_hidden(x)
        x = F.relu(x)

        logits = self.l_output(x)
        # sigmoid probably computing during training

        return logits

class LabelEncoder(nn.Module):
    def __init__(self, z_dim):
        super(LabelEncoder, self).__init__()

        self.l_embed = nn.Embedding(10, 512)
        self.l_input = nn.Linear(512, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.out_mu = nn.Linear(512, z_dim)
        self.out_logvar = nn.Linear(512, z_dim)

    def forward(self, x):

        embed = self.l_embed(x)
        # embed = embed.mean(dim=0)
        # embed = F.relu(embed)

        x = self.l_input(embed)
        x = F.relu(x)

        x = self.l_hidden(x)
        x = F.relu(x)

        logits_mu = self.out_mu(x)
        logits_logvar = self.out_logvar(x)

        return logits_mu, logits_logvar

class LabelDecoder(nn.Module):
    def __init__(self, z_dim):
        super(LabelDecoder, self).__init__()
        self.l_input = nn.Linear(z_dim, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.l_output = nn.Linear(512, 10)

    def forward(self, x):
        x = self.l_input(x)
        x = F.relu(x)

        x = self.l_hidden(x)
        x = F.relu(x)

        logits = self.l_output(x)
        # softmax probably better to use during training function

        return logits


if __name__ == "__main__":

    # image = torch.rand(64, 28, 28)
    # enc = MVAE(20)
    # label = torch.tensor([1,0,0,0,0,0,0,0,0,0])
    # # label = label.unsqueeze(1)
    # # batch_label = label.repeat(5,1)
    # # print(batch_label.size())
    # image = torch.randn(28*28)
    # # gen_image, gen_label, mu, logvar = enc.forward(label_modal=label)
    # gen_image, gen_label, mu, logvar = enc.forward(image_modal=image)
    # # gen_image, gen_label, mu, logvar = enc.forward(image_modal=image, label_modal=label)

    # print("#"*20)
    # print("MU: ")
    # print(mu.size())
    # print(mu)
    # print("LOGVAR: ")
    # print(logvar.size())
    # print(logvar)

    # print("GEN image: ")
    # if gen_image is not None: print(gen_image.size())
    # print(gen_image)
    # print("GEN LAB: ")
    # if gen_label is not None: print(gen_label.size())
    # print(gen_label)

    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms

    transformation = transforms.ToTensor()
    train = datasets.MNIST(root='./mnist', train=True, download=True, transform=transformation)
    test = datasets.MNIST(root='./mnist', train=False, download=True, transform=transformation)

    train_loader = DataLoader(train, batch_size=64, shuffle=False)
    test_loader = DataLoader(test, batch_size=64, shuffle=False)
    model = MVAE(20)
    for x, y in train_loader:
        logit = model(x,y)
