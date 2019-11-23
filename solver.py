from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable

# From Introduction to Deep Learning exercise templates.
# The solver is currently specialized for classification tasks.

class triplet_loss(torch.nn.Module):

    def __init__(self, margin=1.):
        super(triplet_loss, self).__init__()
        self.margin = margin

    def forward(self, text_positive, image_positive, text_anchor, image_anchor, text_negative, image_negative, average=True):

        dist_pos_text = (text_anchor - text_positive).pow(2).sum(1)
        dist_neg_text = (text_anchor - text_negative).pow(2).sum(1)
        loss1 = torch.nn.functional.relu(dist_pos_text - dist_neg_text + self.margin)

        dist_pos_img = (image_anchor - image_positive).pow(2).sum(1)
        dist_neg_img = (image_anchor - image_positive).pow(2).sum(1)
        loss2 = torch.nn.functional.relu(dist_pos_img - dist_neg_img + self.margin)


        dist_pos_text_img = (text_anchor - image_positive).pow(2).sum(1)
        dist_neg_text_img = (text_anchor - image_negative).pow(2).sum(1)
        loss3 = torch.nn.functional.relu(dist_pos_text_img - dist_neg_text_img + self.margin)


        dist_pos_img_text = (image_anchor - text_positive).pow(2).sum(1)
        dist_neg_img_text = (image_anchor - text_negative).pow(2).sum(1)
        loss4 = torch.nn.functional.relu(dist_pos_img_text - dist_neg_img_text + self.margin)

        losses = loss1 + loss2 + loss3 + loss4

        return losses.mean() if size_average else losses.sum()




class Solver(object):
    default_adam_args = {"lr": 1e-2,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=triplet_loss()):
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
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
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

        print("DID THIS RELOAD?")

        print('START TRAIN.')
        
        print("hä?")

        for epoch in range(num_epochs):
            # TRAINING
            print("very start of it")
            model.train()
            print("does it get here?")
            for i, batch in enumerate(train_loader, 1):
                batch = tuple(tensor.to(device) for tensor in batch)
                inputs = batch[:-1]
                targets = batch[-1]

                batch_size = targets.size(0)
                
                optim.zero_grad()
                x_text_positive, x_image_positive, x_text_anchor, x_image_anchor, x_text_negative, x_image_negative = model(inputs)
                print("test if loaded")
                print("before loss")

                loss = self.loss_func(x_text_positive, x_image_positive, x_text_anchor, x_image_anchor, x_text_negative, x_image_negative)
                print("after loss")
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.detach().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss))

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0 e.g. for segmentation
            # targets_mask = targets >= 0
            # train_acc = np.mean((preds == targets)[targets_mask].detach().numpy())
            # self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   0.,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            model.eval()
            for i, batch in enumerate(val_loader, 1):
                batch = tuple(tensor.to(device) for tensor in batch)
                inputs = batch[:-1]
                targets = batch[-1]

                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, targets)
                val_losses.append(loss.detach().numpy())

                #_, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                #targets_mask = targets >= 0
                #scores = np.mean((preds == targets)[targets_mask].detach().numpy())
                #val_scores.append(scores)

            model.train()
            # val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            val_acc, val_loss = 0., np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))

        print('FINISH.')