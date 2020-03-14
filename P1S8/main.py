import data_loader
from resnet import ResNet18
from utils import model_summary
from train import train_model

trainloader, testloader, classes = data_loader.cifar10()
net = ResNet18()
device = model_summary(net, (3, 32, 32))
trainer = train_model(trainloader, testloader, device)
trainer.run_model(net=net, epochs=20, l2=0.00001, gamma=0.1)

