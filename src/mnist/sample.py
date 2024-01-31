import os
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from data import MNISTDataLoader
from utils import load_checkpoint, check_modality_cond



if __name__=="__main__":
  if os.path.basename(os.getcwd()) == 'MVAE':
    model_path = './src/mnist/trained_models/final_best_epoch.pth.tar'
  
  elif os.path.basename(os.getcwd()) == 'src':
    model_path = './trained_models/final_best_epoch.pth.tar'

  condition_on_image = None
  condition_on_text = 6
  n_samples = 64
  
  model, _, checkpoint = load_checkpoint(model_path)
  model.eval()

  mu, std, _, _ = check_modality_cond(condition_on_image, condition_on_text, model)

  samples = torch.randn(n_samples, checkpoint["n_latents"])

  mu         = mu.expand_as(samples)
  std        = std.expand_as(samples)
  samples     = samples.mul(std).add_(mu)

  gen_image  = F.sigmoid(model.image_decoder(samples)).data
  gen_label  = F.log_softmax(model.label_decoder(samples), dim=1).data


  image_data = './sampeled_data'
  if not os.path.exists(image_data):
      os.makedirs(image_data)
  image_path = os.path.join(image_data, 'generated_image.png')
  
  save_image(gen_image.view(n_samples, 1, 28, 28), image_path)

  with open(os.path.join(image_data, 'generated_label.txt'), 'w') as fp:
      label_numpy = gen_label.numpy()
      label_numpy = np.argmax(label_numpy, axis=1).tolist()
      for i, item in enumerate(label_numpy):
          fp.write('Text (%d): %s\n' % (i, item))