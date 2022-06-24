import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#create label map that identifies each type 
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
#create plot figure
figure = plt.figure(figsize=(8, 8))
#organize into cols and rows
cols, rows = 3, 3

#enumerate and paste the image
for i in range(1, cols * rows + 1):
    #create matrix with randints which equals the length of the training data
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #set image and label equal to the index of training_data
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
#plt.show()