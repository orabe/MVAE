import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import MVAE
from data import MNISTDataLoader
from tqdm import tqdm
from torchvision import transforms
import os
from utils import PARAMS, save_checkpoint, load_checkpoint
import torch.optim.lr_scheduler as lr_scheduler

def elbo_mnist(gen_image, image, gen_label, label, mu, logvar,
              lambda_image=PARAMS["lambda_image"], lambda_label=PARAMS["lambda_label"], annealing_factor=1.):

    image_bce, label_bce = 0, 0  # default params
    if gen_image is not None and image is not None:
        image_bce = F.binary_cross_entropy_with_logits(gen_image.view(-1, 1*28*28), image.view(-1, 1*28*28), reduction="none")
        image_bce = torch.sum(image_bce, dim=1)

    if gen_label is not None and label is not None:
       
        label_bce = F.cross_entropy(gen_label, label, reduction="none")

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


    ELBO = torch.mean(lambda_image * image_bce + lambda_label * label_bce 
                      + annealing_factor * KLD)

    return ELBO


if __name__=="__main__":
    
    epochs = PARAMS["epochs"]
    latent_size = PARAMS["latent_size"]
    batch_size = PARAMS["batch_size"]
    beta_anneal = PARAMS["annealing_factor"]
    lr = PARAMS["lr"]

    if os.path.exists('./trained_models/final_best_epoch.pth.tar'):
        model, optimizer, checkpoint = load_checkpoint("./trained_models/final_best_epoch.pth.tar")
        train_loss_list = checkpoint["train_loss_list"]
        val_loss_list = checkpoint["val_loss_list"]
        start_index = len(train_loss_list)
        best_loss = val_loss_list[-1]
        lr = 1e-6

    else:
        model = MVAE(latent_size)
        train_loss_list = []
        val_loss_list = []
        start_index = 0
        best_loss = np.inf       
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = CustomScheduler(optimizer, start_lr=1e-3, end_lr=2e-6, step_size=5)


    transform = transforms.Compose([transforms.ToTensor()])

    mnist_data_loader = MNISTDataLoader(batch_size=batch_size, transform=transform)

    train_dataloader = mnist_data_loader.get_train_data_loader()
    val_dataloader = mnist_data_loader.get_val_data_loader()
    test_dataloader = mnist_data_loader.get_test_data_loader()
    
    def train():

        model.train()

        total_loss = 0.0
        beta = min(1.0, i/beta_anneal)
        for idx, (image, label) in enumerate(tqdm(train_dataloader, desc=f'Epoch {i + 1}/{epochs}, Annealing Factor: {beta:.3f}')):
              
              assert batch_size == len(image) and batch_size == len(label)
              
              optimizer.zero_grad()

              gen_image, _, mu_image, logvar_image = model(image_modal=image, label_modal=None)
              _, gen_label, mu_label, logvar_label = model(image_modal=None, label_modal=label)
              gen_image_joint, gen_label_joint, mu_joint, logvar_joint = model(image_modal=image, label_modal=label)
              
              joint_loss = elbo_mnist(gen_image_joint, image, gen_label_joint, label, mu_joint, logvar_joint, annealing_factor=beta)
              image_loss = elbo_mnist(gen_image, image, None, None, mu_image, logvar_image, annealing_factor=beta)
              label_loss = elbo_mnist(None, None, gen_label, label, mu_label, logvar_label, annealing_factor=beta)

              overall_elbo = joint_loss + image_loss + label_loss

              overall_elbo = joint_loss + image_loss + label_loss

              loss = overall_elbo
              total_loss += loss.item()

              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

              optimizer.step()
    
        average_loss = total_loss / len(train_dataloader)
        train_loss_list.append(average_loss)
        print(f'Epoch {i + 1}/{epochs}, Average Loss: {average_loss:.4f}')



    def validate():

        model.eval()
        total_loss = 0.
        
        for idx, (image, label) in enumerate(val_dataloader):
                    gen_image, _, mu_image, logvar_image = model(image_modal=image, label_modal=None)
                    _, gen_label, mu_label, logvar_label = model(image_modal=None, label_modal=label)
                    gen_image_joint, gen_label_joint, mu_joint, logvar_joint = model(image_modal=image, label_modal=label)
                    
                    joint_loss = elbo_mnist(gen_image_joint, image, gen_label_joint, label, mu_joint, logvar_joint)
                    image_loss = elbo_mnist(gen_image, image, None, None, mu_image, logvar_image)
                    label_loss = elbo_mnist(None, None, gen_label, label, mu_label, logvar_label)

                    overall_elbo = joint_loss + image_loss + label_loss
                    loss = overall_elbo
                    
                    total_loss += loss.item()


        average_loss = total_loss / len(val_dataloader)
        val_loss_list.append(average_loss)
        print(f'Validation for Epoch {i + 1}/{epochs}, Average Loss: {average_loss:.4f}')

        return average_loss

        


    for i in range(start_index, epochs):
        train()
        val_loss = validate()

        is_best = val_loss < best_loss

        if is_best:
          best_loss = val_loss
          save_checkpoint({
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'n_latents': latent_size,
                'optimizer' : optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'val_loss_list': val_loss_list
            }, is_best, folder='./trained_models', filename="final_best_epoch.pth.tar")   
        
        if (i+1) % 2 == 0:
          save_checkpoint({
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'n_latents': latent_size,
                'optimizer' : optimizer.state_dict(),
                'train_loss_list': train_loss_list,
                'val_loss_list': val_loss_list
            }, is_best, folder='./trained_models', filename="epoch_{}.pth.tar".format(i+1))   
