"""import torch
import matplotlib.pyplot as plt
import os

from model import MVAE

def sample_from_mvae(model, n_samples):
    model.eval() 
    with torch.no_grad():
        # Generate random samples from a standard normal distribution
        z_samples = torch.randn(n_samples, model.z_dim)

        # Decode the samples
        gen_img = model.image_decoder(z_samples)
        gen_lbl = model.label_decoder(z_samples)

        return gen_img, gen_lbl

def plot_generated_samples(images, labels):
    _, axs = plt.subplots(4, 4)
    axs = axs.flatten()

    for i in range(images.size(0)):
        img = images[i].view(28, 28).numpy()
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"Label: {torch.argmax(labels[i]).item()}")

    plt.show()

# Load the trained model
# ---------------------------------
# the following works only on colab

# from google.colab import drive
# drive.mount('/content/gdrive/')

save_path = '/content/weights/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# use this before training to save the weights 
# save_interval = 10 # save weights every 10 epochs
model_path = save_path
# ---------------------------------

default_values = {"n_lat_dim": 64, "lr": 1e-3, "n_epochs": 500, "n_first_annealing_epochs": 200}
loaded_model = MVAE(default_values["n_lat_dim"])
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

# Sample from the loaded model
generated_images, generated_labels = sample_from_model(loaded_model, n_samples=16)

# Plot the generated samples
plot_generated_samples(generated_images, generated_labels)

"""


import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np

from model import MVAE



def fetch_mnist_image(label):
    mnist_dataset = datasets.MNIST('./data', train=False, download=True, 
                                   transform=transforms.ToTensor())
    images = mnist_dataset.test_data.numpy()
    labels = mnist_dataset.test_labels.numpy()
    images = images[labels == label]
    image  = images[np.random.choice(np.arange(images.shape[0]))]
    image  = torch.from_numpy(image).float() 
    image  = image.unsqueeze(0)
    return Variable(image, volatile=True)

def fetch_mnist_text(label):
    text = torch.LongTensor([label])
    return Variable(text, volatile=True)


def inference(model, image, label): 
    batch_size = image.size(0) if image is not None else label.size(0)
    
    # initialize the universal prior expert
    gaus_dim = (1, batch_size, model.z_dim)
    mu = Variable(torch.zeros(gaus_dim))
    logvar = Variable(torch.zeros(gaus_dim))

    img_mu, img_logvar = model.image_encoder(image)
    mu = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
    logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

    label_mu, label_logvar = model.label_encoder(label)
    mu = torch.cat((mu, label_mu.unsqueeze(0)), dim=0)
    logvar = torch.cat((logvar, label_logvar.unsqueeze(0)), dim=0)

    # product of experts to combine gaussians
    mu, logvar = model.compute_poe(mu, logvar)
    return mu, logvar


def save_results(n_samples, img_recon, txt_recon, infer_results='infer_results'):
    
    if not os.path.exists(infer_results):
        os.makedirs(infer_results)

    for i in range(n_samples):
        image_path = os.path.join(infer_results, f'generated_image_{i}.png')
        save_image(img_recon[i].view(1, 1, 28, 28), image_path)

    text_path = os.path.join(infer_results, 'generated_text.txt')
    with open(text_path, 'w') as fp:
        txt_recon_np = np.argmax(txt_recon.numpy(), axis=1).tolist()
        for i, item in enumerate(txt_recon_np):
            fp.write(f'Generated Text ({i}): {item}\n')
            

number_to_generate_as_image = 7
number_to_generate_as_text = 7
n_samples = 2 


# Load the trained model
model_path = '/content/weights/model_weights_epoch_0.pth'

default_values = {"n_lat_dim": 64, "lr": 1e-3, "n_epochs": 500, "n_first_annealing_epochs": 200}
loaded_model = MVAE(default_values["n_lat_dim"])
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()


image = fetch_mnist_image(number_to_generate_as_image)
label = fetch_mnist_text(number_to_generate_as_text)
# plt.imshow(image.squeeze())
# print(label)

mu, logvar = inference(loaded_model, image, label)
std = logvar.mul(0.5).exp_()

# sample from uniform gaussian
sample = Variable(torch.randn(n_samples, loaded_model.z_dim))

# sample from particular gaussian by multiplying + adding
mu = mu.expand_as(sample)
std = std.expand_as(sample)
sample = sample.mul(std).add_(mu)

# generate image and text
img_recon = F.sigmoid(loaded_model.image_decoder(sample)).cpu().data
txt_recon = F.log_softmax(loaded_model.label_decoder(sample), dim=1).cpu().data

plt.imshow(img_recon.reshape(n_samples, 28, 28)[1])
txt_recon_np = txt_recon.numpy()
txt_recon_np = np.argmax(txt_recon_np, axis=1).tolist()
txt_recon_np

save_results(n_samples, img_recon, txt_recon)