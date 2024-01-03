import torch
import matplotlib.pyplot as plt
import os

from model import MVAE

def sample_from_model(model, n_samples=16):
    model.eval()
    with torch.no_grad():
        noise = torch.randn(n_samples, model.z_dim)

        generated_images = model.image_decoder(noise)
        generated_labels = model.label_decoder(noise)

    return generated_images, generated_labels

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
