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

        poe_mu, poe_logvar = self.compute_poe(mu, logvar)
    
        z = self.reparameterize(poe_mu, poe_logvar)

        if image_modal is not None:
            gen_image = self.image_decoder(z)
        else:
            gen_image = None
        if label_modal is not None:
            gen_label = self.label_decoder(z)
        else:
            gen_label = None
        

        return gen_image, gen_label, poe_mu, poe_logvar

    def reparameterize(self, poe_mu, poe_logvar):
        std = torch.exp(0.5*poe_logvar)
        
        eps = torch.randn_like(std)
        return eps * std + poe_mu
    
    def prior_expert(self, size):
        normal_mu = torch.zeros(size)
        normal_logvar = torch.zeros(size)
        return normal_mu, normal_logvar

    # The heart of the implementation
    def compute_poe(self, mu, logvar, eps=1e-8):
        # VAE learns logvar, but we need var
        var = torch.exp(logvar) + eps

        # In paper T is inverse of covariance matrix, which is 1/var
        T = 1. / var

        cov_term = torch.sum(T, dim=0)

        poe_cov = 1 / cov_term
        poe_mu = poe_cov * torch.sum(mu * T, dim=0)

        # Get back the logvar to align with learning objective
        poe_logvar = torch.log(poe_cov)

        return poe_mu, poe_logvar


    def prepare_poe(self, image_modal, label_modal):
        if image_modal is None and label_modal is None:
            assert("Both modalities are None. At least one modality must be present.")
        
        n_samples = image_modal.size(0) if image_modal is not None else label_modal.size(0)        
        mu, logvar = self.prior_expert(size=(1,n_samples, self.z_dim))
        
        if image_modal is not None:
            img_mu, img_logvar = self.image_encoder(image_modal)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if label_modal is not None:
            lbl_mu, lbl_logvar = self.label_encoder(label_modal)
            mu     = torch.cat((mu, lbl_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, lbl_logvar.unsqueeze(0)), dim=0)
        
        return mu, logvar




    
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


# if __name__ == "__main__":

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

    # from torch.utils.data import DataLoader
    # from torchvision import datasets
    # import torchvision.transforms as transforms

    # transformation = transforms.ToTensor()
    # train = datasets.MNIST(root='./mnist', train=True, download=True, transform=transformation)
    # test = datasets.MNIST(root='./mnist', train=False, download=True, transform=transformation)

    # train_loader = DataLoader(train, batch_size=100, shuffle=False)
    # test_loader = DataLoader(test, batch_size=100, shuffle=False)
    # model = MVAE(64)
    # for i, (x, y) in enumerate(train_loader):
    #     # print("#"*20)
    #     # print(y)
    #     # print("#"*20)
    #     gen_image_joint, gen_label_joint, mu_joint, logvar_joint = model(image_modal=x, label_modal=y)
