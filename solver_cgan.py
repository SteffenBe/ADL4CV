from datetime import datetime
import numpy as np
import pickle

import torch
from timeit import default_timer as timer

class SolverCGAN(object):
    """Solver for conditional GANs."""

    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, latent_dim, optim=torch.optim.Adam, optim_args_generator={}, optim_args_discriminator={},
                 loss_func=torch.nn.BCELoss()):
        optim_args_generator_merged = self.default_adam_args.copy()
        optim_args_generator_merged.update(optim_args_generator)
        self.optim_args_generator = optim_args_generator_merged
        optim_args_discriminator_merged = self.default_adam_args.copy()
        optim_args_discriminator_merged.update(optim_args_discriminator)
        self.optim_args_discriminator = optim_args_discriminator_merged

        self.optim = optim
        self.loss_func = loss_func

        self.latent_dim = latent_dim
        self.z_for_save = torch.randn(25, self.latent_dim)
        self.condition_for_save = None  # will be filled with values from first batch
        self.instance_id = datetime.utcnow().strftime('%H-%M')
        self._reset_histories()

    def _reset_histories(self):
        self.generator_loss_history = []
        self.discriminator_loss_history = []

    def train(self, generator, discriminator, dataloader, num_epochs=10, log_nth=0, save_nth_batch=0):
        """
        Train a given model with the provided data.

        Inputs:
        - generator: model to train,
        - discriminator: model to train,
        - dataloader: train data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - save_nth_batch: save sample images every n trained batches
        """
        
        self._reset_histories()
        optimizer_G = self.optim(generator.parameters(), **self.optim_args_generator)
        optimizer_D = self.optim(discriminator.parameters(), **self.optim_args_discriminator)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator.to(device)
        discriminator.to(device)
        self.z_for_save = self.z_for_save.to(device)

        iter_per_epoch = len(dataloader)
        print('START TRAIN FOR %d ITERATIONS.' % (iter_per_epoch * num_epochs))
        
        for epoch in range(num_epochs):
            last_start_time = timer()
            # TRAINING
            generator.train()
            discriminator.train()

            # Keep generated images around for the multiple steps of training of the discriminator.
            fake_imgs = None

            for i, batch in enumerate(dataloader):
                batch_index = epoch * len(dataloader) + i
                batch = tuple(tensor.to(device) for tensor in batch)
                batch_size = batch[0].size(0)
                
                condition_inputs, real_imgs = batch
                condition_inputs.fill_(0)

                if type(self.condition_for_save) == type(None):
                    self.condition_for_save = condition_inputs[:25]

                # Adversarial ground truths
                valid = torch.Tensor(batch_size, 1).fill_(1.0).to(device)
                fake = torch.Tensor(batch_size, 1).fill_(0.0).to(device)
                
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()
                # Sample noise as generator input
                z = torch.randn(batch_size, self.latent_dim).to(device)
                gen_input = torch.cat([z, condition_inputs], 1)
                gen_imgs = generator(gen_input)
                # Loss measures generator's ability to fool the discriminator
                g_loss = self.loss_func(discriminator(gen_imgs, condition_inputs), valid)

                g_loss.backward()
                optimizer_G.step()
                self.generator_loss_history.append(g_loss.item())
                fake_imgs = gen_imgs.detach()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                dout_real = discriminator(real_imgs, condition_inputs)
                dout_fake = discriminator(fake_imgs, condition_inputs)
                real_loss = self.loss_func(dout_real, valid)
                fake_loss = self.loss_func(dout_fake, fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                self.discriminator_loss_history.append(d_loss.item())
                if batch_index % save_nth_batch == 0:
                    print("(Saving samples after %d total iterations.)" % batch_index)
                    self._save_samples(generator, batch_index)

                if log_nth and batch_index % log_nth == 0:
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

    def _save_samples(self, generator, n_iterations):
        gen_input = torch.cat([self.z_for_save, self.condition_for_save], 1)
        gen_imgs = generator(gen_input).detach().cpu()
        fname = "images/%s--%05d.png" % (self.instance_id, n_iterations)
        save_image(gen_imgs.permute(0, 3, 1, 2), fname, nrow=5, normalize=True)