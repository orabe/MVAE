import os
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from data import MNISTDataLoader
from utils import load_checkpoint



def fetch_image(label):

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data_loader = MNISTDataLoader(transform=transform, shuffle=True)

    mnist_dataset = mnist_data_loader.get_test_data_loader()

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



if __name__=="__main__":

  condition_on_image = None
  condition_on_text = 5
  n_samples = 64

  model_path = './trained_models/final_best_epoch.pth.tar'
  model, _, checkpoint = load_checkpoint(model_path)
  model.eval()

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

  samples = torch.randn(n_samples, checkpoint["n_latents"])

  mu         = mu.expand_as(samples)
  std        = std.expand_as(samples)
  samples     = samples.mul(std).add_(mu)

  gen_image  = F.sigmoid(model.image_decoder(samples)).data
  gen_label  = F.log_softmax(model.label_decoder(samples), dim=1).data



  save_image(gen_image.view(n_samples, 1, 28, 28),'./imgs/generated_image.png')
  with open('./imgs/generated_label.txt', 'w') as fp:
      label_numpy = gen_label.numpy()
      label_numpy = np.argmax(label_numpy, axis=1).tolist()
      for i, item in enumerate(label_numpy):
          fp.write('Text (%d): %s\n' % (i, item))