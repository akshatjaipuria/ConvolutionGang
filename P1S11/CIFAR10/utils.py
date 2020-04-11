from torchsummary import summary
import torch
import matplotlib.pyplot as plt
import numpy as np


def model_summary(net, size):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = net.to(device)
    print(summary(model, input_size=size))
    return device


def plot_logs(logs):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(logs[0])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(logs[2])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(logs[1])
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(logs[3])
    axs[1, 1].set_title("Test Accuracy")
    plt.show()
    fig.savefig('plot.png')


def denormalize(image, mean, std, out_type='np_array'):
    """Un-normalize a given image,i.e., tensor or numpy array

    Args:
        image: A 3-D numpy array or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        out_type: Out type of the normalized image.
            `np_array` -> then numpy array is returned
            `tensor` ->  then torch tensor is returned.
    """

    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))

    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'np_array':
        return normal_image
    return None


def to_numpy(tensor):
    """tensor -> (C,H,W) to numpy array -> (H,W,C)"""
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def to_tensor(np_array):
    """numpy array -> (H,W,C) to tensor -> (C,H,W)"""
    return torch.Tensor(np.transpose(np_array, (2, 0, 1)))


def best_lr(lr_finder):
    return lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]


def plot_misclassified(img_name, mean, std, misclassified_images, classes):
    plt.figure(figsize=(10, 10))
    num_of_images = len(misclassified_images)
    for index in range(1, num_of_images + 1):
        img = denormalize(misclassified_images[index - 1]["img"], mean, std, )  # Denormalize
        plt.subplot(5, 5, index)
        plt.axis('off')
        plt.imshow(img)
        # plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title("Predicted: %s\nActual: %s" % (
            classes[misclassified_images[index - 1]["pred"]],
            classes[misclassified_images[index - 1]["target"]]))

    plt.tight_layout()
    plt.savefig(img_name)
    plt.show()
