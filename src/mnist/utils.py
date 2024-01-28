import os
import shutil
import torch
import torch.optim as optim
from model import MVAE
from torchvision import transforms, datasets
import numpy as np

PARAMS = {
    "epochs": 500,
    "latent_size": 64,
    "batch_size": 100,
    "lr": 1e-3,
    "annealing_factor": 200,
    "lambda_image": 1.0,
    "lambda_label": 50.0
}

def save_checkpoint(state, is_best, folder='./trained_models', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))

        
def load_checkpoint(file_path, use_cuda=False):
    checkpoint = torch.load(file_path) if use_cuda else \
        torch.load(file_path, map_location=lambda storage, location: storage)
    model = MVAE(checkpoint['n_latents'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, checkpoint

def plot_loss_curve(train=True, model_path="./trained_models/epoch_500.pth.tar"):
    import matplotlib.pyplot as plt

    _, _, checkpoint = load_checkpoint(file_path=model_path)
    if train:
        name = "Training Loss"
        loss_list = checkpoint["train_loss_list"]
    
    elif not train:
        name = "Validation Loss"
        loss_list = checkpoint["val_loss_list"]

    epoch_range = range(1, len(loss_list) + 1)

    plt.plot(epoch_range, loss_list, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("{} Curve".format(name))
    plt.legend()
    plt.savefig("./imgs/{}.png".format(name))
    plt.show()

def fetch_image(label):
    mnist_dataset = datasets.MNIST('./data', train=False, download=True, 
                                    transform=transforms.ToTensor())
    images = mnist_dataset.test_data.numpy()
    labels = mnist_dataset.test_labels.numpy()
    images = images[labels == label]
    image  = images[np.random.choice(np.arange(images.shape[0]))]
    image  = torch.from_numpy(image).float() 
    image  = image.unsqueeze(0)
    return image

def fetch_label(label):
    label = torch.LongTensor([label])
    return label


def check_modality_cond(condition_on_image, condition_on_text, model) :
    image = None
    label = None
    
    if not condition_on_image and not condition_on_text:
        mu = torch.Tensor([0])
        std = torch.Tensor([1])

    elif condition_on_image and not condition_on_text:
        image = fetch_image(condition_on_image)
        tmp_mu, tmp_logvar = model.prepare_poe(image_modal=image, label_modal=None)

        mu, logvar = model.compute_poe(tmp_mu, tmp_logvar)
        std = logvar.mul(0.5).exp_()

    elif condition_on_text and not condition_on_image:
        label = fetch_label(condition_on_text)

        tmp_mu, tmp_logvar = model.prepare_poe(label_modal=label, image_modal=None)

        mu, logvar = model.compute_poe(tmp_mu, tmp_logvar)

        std = logvar.mul(0.5).exp_()

    elif condition_on_text and condition_on_image:
        image = fetch_image(condition_on_image)
        label = fetch_label(condition_on_text)

        tmp_mu, tmp_logvar = model.prepare_poe(image_modal=image, label_modal=label)

    mu, logvar = model.compute_poe(tmp_mu, tmp_logvar)
    std = logvar.mul(0.5).exp_()
    
    return mu, std, image, label
        
    
      
# if __name__=="__main__":
#     plot_loss_curve(train=True)