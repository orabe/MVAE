import torch
import torch.nn.functional as F
import os

from utils import load_checkpoint, check_modality_cond

from eval_utils import (
    bernoulli_log_pdf,
    categorical_log_pdf,
    gaussian_log_pdf,
    unit_gaussian_log_pdf,
    log_mean_exp)

def log_joint_estimate(recon_image, image, recon_label, label, z, mu, logvar):
	r"""Estimate log p(x,y).

	@param recon_image: torch.Tensor (batch size x # samples x 784)
						reconstructed means on bernoulli
	@param image: torch.Tensor (batch size x 784)
						original observed image
	@param recon_label: torch.Tensor (batch_size x # samples x n_class)
						reconstructed logits
	@param label: torch.Tensor (batch_size)
 						original observed labels
	@param z: torch.Tensor (batch_size x # samples x z dim)
						samples drawn from variational distribution
	@param mu: torch.Tensor (batch_size x # samples x z dim)
						means of variational distribution
	@param logvar: torch.Tensor (batch_size x # samples x z dim)
						log-variance of variational distribution
	"""
	batch_size, n_samples, z_dim = z.size()
	input_dim = image.size(1)
	label_dim = recon_label.size(2)
	image = image.unsqueeze(1).repeat(1, n_samples, 1)
	label = label.unsqueeze(1).repeat(1, n_samples)

	z2d = z.view(batch_size * n_samples, z_dim)
	mu2d = mu.view(batch_size * n_samples, z_dim)
	logvar2d = logvar.view(batch_size * n_samples, z_dim)
	recon_image_2d = recon_image.view(batch_size * n_samples, input_dim)
	image_2d = image.view(batch_size * n_samples, input_dim)
	recon_label_2d = recon_label.view(batch_size * n_samples, label_dim)
	label_2d = label.view(batch_size * n_samples)

	log_p_x_given_z_2d = bernoulli_log_pdf(image_2d, recon_image_2d)
	log_p_y_given_z_2d = categorical_log_pdf(label_2d, recon_label_2d)
	log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
	log_p_z_2d = unit_gaussian_log_pdf(z2d)

	log_weight_2d = log_p_x_given_z_2d + log_p_y_given_z_2d + \
	log_p_z_2d - log_q_z_given_x_2d
	log_weight = log_weight_2d.view(batch_size, n_samples)

	# need to compute normalization constant for weights
	# i.e. log ( mean ( exp ( log_weights ) ) )
	log_p = log_mean_exp(log_weight, dim=1)
	return -torch.mean(log_p)



def log_marginal_estimate(recon_image, image, z, mu, logvar):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.

    @param recon_image: torch.Tensor (batch size x # samples x 784)
                        reconstructed means on bernoulli
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size, n_samples, z_dim = z.size()
    input_dim = image.size(1)
    image = image.unsqueeze(1).repeat(1, n_samples, 1)

    z2d = z.view(batch_size * n_samples, z_dim)
    mu2d = mu.view(batch_size * n_samples, z_dim)
    logvar2d = logvar.view(batch_size * n_samples, z_dim)
    recon_image_2d = recon_image.view(batch_size * n_samples, input_dim)
    image_2d = image.view(batch_size * n_samples, input_dim)

    log_p_x_given_z_2d = bernoulli_log_pdf(image_2d, recon_image_2d)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return -torch.mean(log_p)


if __name__=="__main__":

	if os.path.basename(os.getcwd()) == 'MVAE':
		model_path = './src/mnist/trained_models/final_best_epoch.pth.tar'

	elif os.path.basename(os.getcwd()) == 'src':
		model_path = './trained_models/final_best_epoch.pth.tar'

	condition_on_image = 5
	condition_on_text = 5
	n_samples = 64

	model, _, checkpoint = load_checkpoint(model_path)
	model.eval()

	mu, std, image, label = check_modality_cond(condition_on_image, condition_on_text, model)

	samples = torch.randn(n_samples, checkpoint["n_latents"])

	mu         = mu.expand_as(samples)
	std        = std.expand_as(samples)
	samples     = samples.mul(std).add_(mu)

	gen_image  = F.sigmoid(model.image_decoder(samples)).data
	gen_label  = F.log_softmax(model.label_decoder(samples), dim=1).data

	# The following steps can be later surpressed by rewriting the `log_joint_estimate` function.
	batch_size =  image.size(0)
	n_class = gen_label.size(1)
	image = image.view(batch_size, 784)  
	gen_label = gen_label.view(batch_size, n_samples, n_class)
	samples = samples.view(batch_size, n_samples, model.z_dim)

	log_joint_est = log_joint_estimate(gen_image, image, gen_label, label, samples, mu, std)
	log_marginal_est = log_marginal_estimate(gen_image, image, samples, mu, std)
	
	print(log_joint_est)
	print(log_marginal_est)
 
