from random import shuffle
import numpy as np
import pickle

import torch
from torch.autograd import Variable
from sklearn.neighbors import KNeighborsClassifier
from timeit import default_timer as timer

class SolverModifier(object):
    default_adam_args = {"lr": 1e-2, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []
        self.model_snapshots = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, snapshot_interval=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN FOR %d ITERATIONS.' % (iter_per_epoch * num_epochs))

        last_start_time = timer()
        for epoch in range(num_epochs):
            # TRAINING
            model.train()
            for i, batch in enumerate(train_loader):
                batch_index = epoch * len(train_loader) + i
                batch = tuple(tensor.to(device) for tensor in batch)
                inputs = batch[:-1]
                targets = batch[-1]

                batch_size = targets.size(0)
                
                optim.zero_grad()
                pred = model(*inputs)
                loss = self.loss_func(pred, targets)
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.item())
                if log_nth and batch_index % log_nth == 0 and batch_index > 0:
                    now = timer()
                    time_elapsed = now - last_start_time
                    iter_per_second = log_nth / time_elapsed
                    last_start_time = now

                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f (%.1f iter/s, %.1f samples/s)' % \
                        (batch_index,
                         iter_per_epoch * num_epochs,
                         train_loss,
                         iter_per_second, iter_per_second * batch_size))

            # VALIDATION
            val_losses = []
            model.eval()
            for i, batch in enumerate(val_loader, 1):
                batch = tuple(tensor.to(device) for tensor in batch)
                inputs = batch[:-1]
                targets = batch[-1]
                pred = model(*inputs)
                loss = self.loss_func(pred, targets)
                val_losses.append(loss.item())
            model.train()

            mean_val_loss = np.mean(val_losses)
            self.val_loss_history.append(mean_val_loss)
            
            mean_train_loss = np.mean(self.train_loss_history[-iter_per_epoch:])
            print('[Epoch %d/%d] TRAIN loss: %.3f; VAL loss: %.3f' % (
                        epoch + 1, num_epochs,
                        mean_train_loss, mean_val_loss))
            
            if snapshot_interval > 0 and (num_epochs - epoch) % snapshot_interval == 1:
                print('(Saving snapshot)')
                self._create_model_snapshot(model)

        print('FINISH.')
    
    def _create_model_snapshot(self, model):
        snapshot = pickle.loads(pickle.dumps(model.state_dict()))
        self.model_snapshots.append(snapshot)