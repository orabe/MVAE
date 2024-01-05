import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Multinomial, Bernoulli
from model import MVAE
from data import MNISTDataLoader
from tqdm import tqdm
import os
import argparse
import sys
import shutil
# import click
from torchvision import transforms


def multi_elbo(
        gen_image,
        gen_label,
        image,
        label,
        mu,
        logvar
):
    reconstruction_image = single_elbo(gen_image, image, logvar, mu, modal_type="image")
    reconstruction_label = single_elbo(gen_label, label, logvar, mu, modal_type="label")

    kld = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)

    lambda_image = 0.0
    lambda_label = 0.0
    beta_weights = 0.0

    elbo = lambda_image * reconstruction_image + lambda_label * reconstruction_label + beta_weights * kld
    return torch.mean(elbo)

def single_elbo(
        gen_modal, 
        modal, 
        logvar, 
        mu, 
        modal_type,
        eps=1e-6
):
    if modal_type == "image":
        # reconstruction_error = bernoulli_likelihood(gen_modal, modal)
        modal = modal.view(-1, 1*28*28)
        print("IMAGE MODAL VIEW")
        print(modal.size())
        gen_modal = gen_modal.view(-1, 1*28*28)
        print("RECON IMAGE MODAL VIEW")
        print(gen_modal.size())
        # reconstruction_error = (torch.clamp(gen_modal, 0) - gen_modal * modal + torch.log(1 + torch.exp(-torch.abs(gen_modal))))
        reconstruction_error = F.binary_cross_entropy_with_logits(gen_modal.view(-1, 1*28*28), modal.view(-1, 1*28*28))
        print(reconstruction_error.size())
    elif modal_type == "label":
        # reconstruction_error = multinomial_likelihood(gen_modal, modal)
        log_recon = F.log_softmax(gen_modal+ eps, dim=1)
        one_hot = log_recon.data.new(log_recon.size()).zero_()
        one_hot = one_hot.scatter(1, modal.unsqueeze(1), 1)

        loss = one_hot*log_recon
        # reconstruction_error = -loss
        reconstruction_error = F.cross_entropy(gen_modal, modal)
        # print(reconstruction_error.size())
        



    # kld = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
    kld = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)
    print("ERROR RECON")
    print(modal_type)
    print(reconstruction_error.size())
    print(reconstruction_error)
    print("KLD")
    print(kld.size())
    print(kld.item())
    lambda_weight = 0.0
    beta_weight = 0.0
    
    elbo = lambda_weight * reconstruction_error + beta_weight * kld

    return torch.mean(elbo)

def elbo_mnist(rec_image, true_image, rec_label, true_label, mu, logvar, beta, lambda_img=1., lambda_label=50., eps=1e-6):

    image_err, label_err = 0,0
    
    if rec_image is not None:
        image_err = F.binary_cross_entropy_with_logits(rec_image.view(-1, 1*28*28), true_image.view(-1, 1*28*28))
    
    if rec_label is not None:
        log_recon = F.log_softmax(rec_label + eps, dim=1)
        one_hot = log_recon.data.new(log_recon.size()).zero_()
        one_hot = one_hot.scatter(1, true_label.unsqueeze(1), 1)
        
        label_err = F.cross_entropy(rec_label, true_label)
    
    KL = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)

    loss = torch.mean(lambda_img*image_err + lambda_label*label_err + beta*KL)

    return loss


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))

        
def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__=="__main__":
    import os
    import argparse
    import sys
    
    epochs = 500
    latent_size = 64
    batch_size = 100
    lr = 1e-6

    if not os.path.isdir('./trained_models'):
        os.makedirs('./trained_models')

    train_dataloader = MNISTDataLoader(batch_size=batch_size, train=True).get_data_loader()
    test_dataloader = MNISTDataLoader(batch_size=batch_size, train=False).get_data_loader()

    transform = transforms.Compose([transforms.ToTensor()])

    # Training DataLoader
    train_data_loader = MNISTDataLoader(transform=transform, train=True, validation=False)

    # Validation DataLoader
    val_data_loader = MNISTDataLoader(transform=transform, train=True, validation=True)

    test_data_loader = MNISTDataLoader(transform=transform, train=False, validation=False)

    model = MVAE(latent_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def train():


        model.train()

        for i in range(epochs):
            total_loss = 0.0
            with tqdm(total=len(train_dataloader), desc=f'Epoch {i+1}/{epochs}', unit="batch") as pbar:
                for idx, (image, label) in enumerate(train_dataloader):
                    assert batch_size == len(image) and batch_size == len(label)
                    beta = min(1.0, epochs/200)
                    optimizer.zero_grad()
                    print(image.size())
                    # label = torch.tensor(label).to(torch.int64)
                    label = label.clone().detach().to(torch.int64)
                    # print(f'Image is of type {type(image)} and has size {image.size()}')
                    # print(f'Label is of type {type(label)} and has size {label.size()} and consists of {label}')
                    gen_image, _, mu_image, logvar_image = model(image_modal=image, label_modal=None)
                    _, gen_label, mu_label, logvar_label = model(image_modal=None, label_modal=label)
                    print("HERE")
                    gen_image_joint, gen_label_joint, mu_joint, logvar_joint = model(image_modal=image, label_modal=label)

                    joint_loss = elbo_mnist(gen_image_joint, image, gen_label_joint, label, mu_joint, logvar_joint, beta)
                    image_loss = elbo_mnist(gen_image, image, None, None, mu_image, logvar_image, beta)
                    label_loss = elbo_mnist(None, None, gen_label, label, mu_label, logvar_label, beta)

                    overall_elbo = joint_loss + image_loss + label_loss

                    loss = -overall_elbo
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()


                    pbar.set_postfix({"Loss": total_loss})
                    pbar.update(1)

    def validate():
        model.eval()
        loss_list = []
        for idx, (image, label) in enumerate(val_data_loader):
                    gen_image, _, mu_image, logvar_image = model(image_modal=image, label_modal=None)
                    _, gen_label, mu_label, logvar_label = model(image_modal=None, label_modal=label)
                    print("HERE")
                    gen_image_joint, gen_label_joint, mu_joint, logvar_joint = model(image_modal=image, label_modal=label)
                    
                    joint_loss = elbo_mnist(gen_image_joint, image, gen_label_joint, label, mu_joint, logvar_joint, beta)
                    image_loss = elbo_mnist(gen_image, image, None, None, mu_image, logvar_image, beta)
                    label_loss = elbo_mnist(None, None, gen_label, label, mu_label, logvar_label, beta)

                    overall_elbo = joint_loss + image_loss + label_loss
                    loss_list.append(overall_elbo)

        return np.mean(loss_list)

        


    best_loss = sys.maxint
    for i in range(epochs):
        train(
            epochs,
            latent_size,
            batch_size,
            lr
        )
    
        test_loss = validate()

        is_best = test_loss < best_loss
        if is_best:
             best_loss = test_loss
        

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'n_latents': latent_size,
            'optimizer' : optimizer.state_dict(),
        }, is_best, folder='./trained_models')   