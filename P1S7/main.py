from model import data_loader
from model.net import Net
from utils import model_summary
from train import train_model

trainloader, testloader, classes = data_loader.cifar10()
net = Net()
device = model_summary(net, (3, 32, 32))
trainer = train_model(trainloader, testloader, device)
trainer.run_model(net=net)
