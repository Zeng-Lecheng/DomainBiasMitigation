import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from models import basenet
from models import dataloader
from models.cifar_core import CifarModel
import utils

class CifarDomainIndependent(CifarModel):
    def __init__(self, opt):
        super(CifarDomainIndependent, self).__init__(opt)

    def set_network(self, opt):
        self.network = basenet.ResNet18_duo(num_classes=opt['output_dim']).to(self.device)

    def forward(self, x):
        return self.network(x)  # out_1, out_2, feature

    def _train(self, loader):
        self.network.train()
        self.adjust_lr()

        train_loss = 0
        total = 0
        correct = 0
        for i, (images, targets) in enumerate(loader):
            self.optimizer.zero_grad()
            out_1, out_2, _ = self.forward(images)
            loss = self._criterion(out_1, out_2, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_result = {
                'loss': loss.item(),
            }
            self.log_result('Train iteration', train_result, len(loader) * self.epoch + i)

            if self.print_freq and (i % self.print_freq == 0):
                print(f'Training epoch {self.epoch}: [{i + 1}|{len(loader)}], loss: {loss.item()}')

        self.epoch += 1
    
    def _criterion(self, out_1, out_2, target):
        class_num = out_1.size(1)
        out_1 = F.pad(out_1, (0, class_num, 0, 0), 'constant', out_1.mean().item())   # fill 0 to the right
        logprob_first_half = F.log_softmax(out_1, dim=1)
        out_1_loss = F.nll_loss(logprob_first_half, target)

        out_2 = F.pad(out_2, (class_num, 0, 0, 0), 'constant', out_2.mean().item())   # fill 0 to the left
        logprob_second_half = F.log_softmax(out_2, dim=1)
        out_2_loss = F.nll_loss(logprob_second_half, target)

        return out_1_loss + out_2_loss
        
    def _test(self, loader, test_on_color=True):
        """Test the model performance"""
        
        self.network.eval()

        total = 0
        correct = 0
        test_loss = 0
        output_list = []
        feature_list = []
        target_list = []
        with torch.no_grad():
            for i, (images, targets) in enumerate(loader):
                images, targets = images.to(self.device), targets.to(self.device)
                out_1, out_2, features = self.forward(images)
                loss = self._criterion(out_1, out_2, targets)
                test_loss += loss.item()
                outputs = torch.cat((out_1, out_2), dim=1)

                output_list.append(outputs)
                feature_list.append(features)
                target_list.append(targets)
                
        outputs = torch.cat(output_list, dim=0)
        features = torch.cat(feature_list, dim=0)
        targets = torch.cat(target_list, dim=0)
        
        accuracy_conditional = self.compute_accuracy_conditional(outputs, targets, test_on_color)
        accuracy_sum_out = self.compute_accuracy_sum_out(outputs, targets)
        
        test_result = {
            'accuracy_conditional': accuracy_conditional,
            'accuracy_sum_out': accuracy_sum_out,
            'outputs': outputs.cpu().numpy(),
            'features': features.cpu().numpy()
        }
        return test_result
    
    def compute_accuracy_conditional(self, outputs, targets, test_on_color): # eq. 7
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        
        class_num = outputs.shape[1] // 2
        if test_on_color:
            outputs = outputs[:, :class_num]
        else:
            outputs = outputs[:, class_num:]
        predictions = np.argmax(outputs, axis=1)
        accuracy = (predictions == targets).mean() * 100.
        return accuracy
    
    def compute_accuracy_sum_out(self, outputs, targets):
        outputs = outputs.cpu().numpy()
        targets = targets.cpu().numpy()
        
        class_num = outputs.shape[1] // 2
        predictions = np.argmax(outputs[:, :class_num] + outputs[:, class_num:], axis=1)    # Eq.9
        accuracy = (predictions == targets).mean() * 100.
        return accuracy

    def test(self):
        # Test and save the result
        state_dict = torch.load(os.path.join(self.save_path, 'ckpt.pth'))
        self.load_state_dict(state_dict)
        test_color_result = self._test(self.test_color_loader, test_on_color=True)
        test_gray_result = self._test(self.test_gray_loader, test_on_color=False)
        utils.save_pkl(test_color_result, os.path.join(self.save_path, 'test_color_result.pkl'))
        utils.save_pkl(test_gray_result, os.path.join(self.save_path, 'test_gray_result.pkl'))
        
        # Output the classification accuracy on test set for different inference
        # methods
        info = ('Test on color images accuracy conditional: {}\n' 
                'Test on color images accuracy sum out: {}\n'
                'Test on gray images accuracy conditional: {}\n'
                'Test on gray images accuracy sum out: {}\n'
                .format(test_color_result['accuracy_conditional'],
                        test_color_result['accuracy_sum_out'],
                        test_gray_result['accuracy_conditional'],
                        test_gray_result['accuracy_sum_out']))
        utils.write_info(os.path.join(self.save_path, 'test_result.txt'), info)
    
    