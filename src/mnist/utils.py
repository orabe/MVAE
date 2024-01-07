import os
import shutil
import torch
import torch.optim as optim
from model import MVAE

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
    # if is_best:
    #     shutil.copyfile(os.path.join(folder, filename),
    #                     os.path.join(folder, 'model_best.pth.tar'))

        
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
    
if __name__=="__main__":
    plot_loss_curve(train=True)