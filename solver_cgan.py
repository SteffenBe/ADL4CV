from datetime import datetime
import numpy as np
import os

import torch
from torchvision.utils import save_image
from timeit import default_timer as timer

class SolverCGAN(object):
    """Solver for conditional GANs."""

    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, latent_dim, optim=torch.optim.Adam, optim_args_generator={}, optim_args_discriminator={},
                 loss_func=torch.nn.BCELoss(),
                 condition_input_for_save=None, z_samples=5):
        optim_args_generator_merged = self.default_adam_args.copy()
        optim_args_generator_merged.update(optim_args_generator)
        self.optim_args_generator = optim_args_generator_merged
        optim_args_discriminator_merged = self.default_adam_args.copy()
        optim_args_discriminator_merged.update(optim_args_discriminator)
        self.optim_args_discriminator = optim_args_discriminator_merged

        self.optim = optim
        self.loss_func = loss_func

        self.instance_id = datetime.utcnow().strftime('%H-%M')
        self.latent_dim = latent_dim
        # For saving the debug image during training.
        if type(condition_input_for_save) == type(None):
            self.condition_input_for_save = None  # will be filled with values from first batch
            self.z_for_save = torch.randn(z_samples, self.latent_dim)
        else:
            self.condition_input_for_save = condition_input_for_save
            self.z_for_save = torch.randn(z_samples, self.latent_dim)
        
        self._reset_histories()

    def _reset_histories(self):
        self.generator_loss_history = []
        self.discriminator_loss_history = []
    
    def _find_mismatching_indices_in_batch(self, batch_labels, verbose=False):
        """For each label in a batch, searches for an entry with a different label and returns its index.
        Since labels can have different numbers of occurences, the returned indices are likely to contain duplicates.

        This is pretty slow, but for our purposes still sufficient (measured around 2.2 ms for a batch size of 100).
        
        Input:  tensor of shape (N, 1) containing class labels
        Output: tensor of shape (N, 1) containing indices of mismatching class labels
        """

        labels = batch_labels.detach().cpu().numpy()  # don't read it from GPU in the loop
        batch_size = labels.shape[0]
        indices = torch.randint(batch_size, size=(batch_size,))
        n_repicks = 0
        for i in range(batch_size):
            # Pick a new index until we found a different class
            while labels[indices[i]] == labels[i]:
                indices[i] = torch.randint(batch_size, size=(1,))
                n_repicks += 1
        
        if verbose: print("n_repicks:", n_repicks)
        return indices

    def train(self, generator, discriminator, dataloader, num_epochs=10, log_nth=0, save_nth_batch=0):
        """
        Train a given model with the provided data.

        Inputs:
        - generator: model to train
        - discriminator: model to train
        - dataloader: train data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: output losses and stats every n trained batches
        - save_nth_batch: save sample images every n trained batches
        """
        
        self._reset_histories()
        optimizer_G = self.optim(generator.parameters(), **self.optim_args_generator)
        optimizer_D = self.optim(discriminator.parameters(), **self.optim_args_discriminator)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator.to(device)
        discriminator.to(device)

        iter_per_epoch = len(dataloader)
        print('START TRAIN FOR %d ITERATIONS.' % (iter_per_epoch * num_epochs))
        
        last_start_time = timer()
        for epoch in range(num_epochs):
            # TRAINING
            generator.train()
            discriminator.train()
            
            # Keep last generated images around in case there are multiple steps of training of the discriminator.
            fake_imgs = None

            for i, batch in enumerate(dataloader):
                batch_index = epoch * len(dataloader) + i
                batch = tuple(tensor.to(device) for tensor in batch)
                batch_size = batch[0].size(0)
                
                condition_inputs, real_imgs, labels = batch

                if type(self.condition_input_for_save) == type(None):
                    self.condition_input_for_save = condition_inputs[:5]

                # Adversarial ground truths. Set target for real images to 0.9 instead of 1
                # to apply "label smoothing" (prevent overconfidence of discriminator).
                valid = torch.Tensor(batch_size, 1).fill_(0.9).to(device)
                fake = torch.Tensor(batch_size, 1).fill_(0.0).to(device)
                
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # Sample noise as generator input
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_input = torch.cat([z, condition_inputs], 1)
                gen_imgs = generator(gen_input)
                fake_imgs = gen_imgs.detach()

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.loss_func(discriminator(gen_imgs, condition_inputs), valid)

                g_loss.backward()
                optimizer_G.step()
                self.generator_loss_history.append(g_loss.item())

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # For each sample in this batch, find ones that do not match.
                wrong_condition_inputs = condition_inputs[self._find_mismatching_indices_in_batch(labels)]

                # Measure discriminator's ability to classify real from generated samples
                d_out_real = discriminator(real_imgs, condition_inputs)
                d_out_fake = discriminator(fake_imgs, condition_inputs)
                # Measure discriminator's ability to detect wrongly assigned conditioning information.
                d_out_wrong = discriminator(real_imgs, wrong_condition_inputs)

                real_loss = self.loss_func(d_out_real, valid)
                fake_loss = self.loss_func(d_out_fake, fake)
                wrong_loss = self.loss_func(d_out_wrong, fake)
                d_loss = real_loss + (fake_loss + wrong_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                self.discriminator_loss_history.append(d_loss.item())
                if batch_index % save_nth_batch == 0:
                    print("(Saving samples after %d total iterations.)" % batch_index)
                    self._save_samples(generator, batch_index, device)

                if log_nth and batch_index % log_nth == 0 and batch_index > 0:
                    now = timer()
                    time_elapsed = now - last_start_time
                    iter_per_second = log_nth / time_elapsed
                    last_start_time = now

                    mean_d_loss = np.mean(self.discriminator_loss_history[-log_nth:])
                    mean_g_loss = np.mean(self.generator_loss_history[-log_nth:])
                    print("[Iteration %d/%d] [D loss: %f] [G loss: %f] (%.1f iter/s, %.1f samples/s)" % (
                        batch_index, iter_per_epoch * num_epochs,
                        mean_d_loss.item(), mean_g_loss.item(),
                        iter_per_second, iter_per_second * batch_size))
            
            mean_d_loss = np.mean(self.discriminator_loss_history[-iter_per_epoch:])
            mean_g_loss = np.mean(self.generator_loss_history[-iter_per_epoch:])
            print("[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch+1, num_epochs, mean_d_loss.item(), mean_g_loss.item()))

        print('FINISH.')
        generator.eval()
        discriminator.eval()

    def _save_samples(self, generator, n_iterations, device):
        num_conditions = self.condition_input_for_save.shape[0]
        num_z = self.z_for_save.shape[0]

        gen_input = torch.cat([torch.repeat_interleave(self.z_for_save, repeats=num_conditions, dim=0),
                                self.condition_input_for_save.repeat(num_z, 1)], 1)

        generator.eval()
        gen_imgs = generator(gen_input.to(device)).detach().cpu()
        generator.train()

        os.makedirs("images", exist_ok=True)
        fname = "images/%s--%05d.png" % (self.instance_id, n_iterations)
        save_image(gen_imgs.permute(0, 3, 1, 2), fname, nrow=num_conditions, normalize=True)