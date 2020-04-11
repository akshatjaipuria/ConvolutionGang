from tqdm import tqdm
import torch
import torch.optim as optim
from torch import nn
import numpy as np

if __name__ == '__main__':
    "the rest of the codes in train.py"


class train_model:
    def __init__(self, trainloader, testloader, device):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.misclassified_images = []
        self.lr_hist=[]

    def train(self, model, device, train_loader, optimizer, loss_func, epoch, l1, scheduler=None):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = loss_func(y_pred, target)

            # L1 regularization
            if l1 > 0:
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += torch.sum(abs(param))
                factor = l1
                loss += factor * reg_loss

            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
            self.train_acc.append(100 * correct / processed)
            if scheduler is not None:
                scheduler.step()

    def test(self, model, device, test_loader, loss_func, find_misclassified=False):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_func(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                is_correct = pred.eq(target.view_as(pred))
                if find_misclassified:
                    misclassified_inds = (is_correct == 0).nonzero()[:, 0]
                    for mis_ind in misclassified_inds:
                        if len(self.misclassified_images) == 25:
                            break
                        self.misclassified_images.append({
                            "target": target[mis_ind].cpu().numpy(),
                            "pred": pred[mis_ind][0].cpu().numpy(),
                            "img": np.transpose(data[mis_ind].cpu().clone().numpy(), (1, 2, 0))
                        })
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))
        return test_loss

    def run_model(self, net, lr=0.01, epochs=10, l1=0, l2=0, max_lr=0.1, sch=False):
        """
        l1 and l2 hold the lambda values for the respective regularization.
        l1 - Lasso Regularization
        l2 - Ridge Regularization
        """
        model = net.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=l2)
        if sch:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                            steps_per_epoch=len(self.trainloader), epochs=epochs,
                                                            pct_start=(5/24), cycle_momentum=True, div_factor=10,
                                                            final_div_factor=1)
        loss_func = nn.CrossEntropyLoss()
        for epoch in range(1, epochs + 1):
            print("EPOCH:", epoch)
            if sch:
                print("LR:",scheduler.get_lr())
                self.lr_hist.append(scheduler.get_lr())
                self.train(model, self.device, self.trainloader, optimizer, loss_func, epoch, l1, scheduler)
            else:
                self.train(model, self.device, self.trainloader, optimizer, loss_func, epoch, l1,)

            if epoch != epochs:
                val_loss = self.test(model, self.device, self.testloader, nn.CrossEntropyLoss(reduction='sum'))
            else:
                val_loss = self.test(model, self.device, self.testloader, nn.CrossEntropyLoss(reduction='sum'),
                                     find_misclassified=True)
            # if sch:
            #     scheduler.step()
        return [self.train_losses, self.test_losses, self.train_acc, self.test_acc], self.misclassified_images, self.lr_hist
