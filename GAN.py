import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class Discriminator(nn.Module):
    """
    Attributes

    conv1 : nn.Conv2d
        First convolutional layer (input: 1 channel, output: 64 channels)
    conv2 : nn.Conv2d
        Second convolutional layer (input: 64 channels, output: 128 channels)
    fc1 : nn.Linear
        First fully connected layer (input: 512 features, output: 512 features)
    fc2 : nn.Linear
        Second fully connected layer (input: 512 features, output: 1 feature)
    sigmoid : nn.Sigmoid
        Sigmoid activation for binary classification

    Methods

    forward(x)
        Defines the forward pass of the model
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Input: [batch_size, 1, 28, 28]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)  # Output: [batch_size, 128, 7, 7]
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters
            x : tensor
            Input tensor with shape [batch_size, 1, 28, 28]

        Returns
            tensor
            Output tensor with shape [batch_size, 1]
        """
        x = nn.LeakyReLU(0.2)(self.conv1(x)) # Computation to find edges. Different kernels find different angles of edges
        x = nn.MaxPool2d(2)(x) # Keeps only the maximum value from each subregion, to reduce overfitting and complexity
        x = nn.LeakyReLU(0.2)(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1) # Flattening to 1D
        x = nn.LeakyReLU(0.2)(self.fc1(x)) # Fully connected layer for classification
        x = self.fc2(x)
        x = self.sigmoid(x) # For binary classification
        return x

class Generator(nn.Module):
    """
    Attributes

    init_size : int
        Size of the image
    l1 : nn.Sequential
        Linear layer followed by Batch Normalization and Upsampling
    conv_blocks : nn.Sequential
        Series of convolutional layers, batch normalization, upsampling, and activations

    Methods

    forward(z)
        Defines the forward pass of the model
    """
    def __init__(self, noise_dim):
        """
            Parameters

            noise_dim : int
            Dimension of the noise vector used to generate images
        """
        super(Generator, self).__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(noise_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        """
            Defines the forward pass of the model.

            Parameters
                z : tensor
                Input noise tensor with shape [batch_size, noise_dim]

            Returns
                tensor
                Generated image tensor with shape [batch_size, 1, 28, 28]
        """
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


noise_dim = 100
generator = Generator(noise_dim)

discriminator = Discriminator()

# Define a transformation to normalize the data
# This makes it suitable for the neural networks
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
mnist_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# Filter the dataset to keep only the zeroes and ones
indices = np.where((mnist_data.targets == 0) | (mnist_data.targets == 1))[0]
binary_data = Subset(mnist_data, indices)

# Create a DataLoader for the filtered dataset
dataloader = DataLoader(binary_data, batch_size=64, shuffle=True)


# Plots an image, given a tensor
def show_image(img):
    # Unnormalize img
    img = img / 2 + 0.5
    # Convert the tensor to a NumPy array
    npimg = img.detach().cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def initial_train_d(num_epochs, model, criterion, optimizer):
    """
        Function to initially train the discriminator model.

        Parameters

        num_epochs : int
            The number of epochs to train the model
        model : nn.Module
            The discriminator model to be trained
        criterion : nn.Module
            The loss function used to calculate the error
        optimizer : torch.optim.Optimizer
            The optimizer used to update model parameters

        Returns

        None
    """
    for epoch in range(num_epochs):
        for image, label in dataloader:
            optimizer.zero_grad()
            outputs = model(image)
            label = label.view(-1, 1).float()
            #print(outputs)
            #print(label)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()


def train_d(optimizer, noise, real_batch):
    """
        Function to train the discriminator in the main loop

        Parameters

        optimizer : torch.optim.Optimizer
            The optimizer used to update model parameters
        noise : tensor
            Noise tensor for generator to use
        real_batch : tensor
            Real images for loss function calculation

        Returns

        None
    """
    optimizer.zero_grad()
    loss = discriminator_loss(noise, real_batch)
    loss.backward()
    optimizer.step()


def train_g(input_dis, optimizer):
    """
        Function to train the generator in the main loop

        Parameters

        optimizer : torch.optim.Optimizer
            The optimizer used to update model parameters
        input_dis : tensor
            Noise for image generation

        Returns

        None
    """
    optimizer.zero_grad()
    loss = generator_loss(input_dis)
    loss.backward()
    optimizer.step()


def generator_loss(z):
    # Generate fake images
    fake_images = generator.forward(z)
    show_image(torchvision.utils.make_grid(fake_images))
    # Discriminator output for fake image
    fake_output = discriminator(fake_images)
    # Calculate loss
    loss = -torch.log(fake_output).mean()
    #print("g:", loss)
    return loss

def discriminator_loss(z, x):
    # Generate fake images
    fake_images = generator.forward(z)
    # Discriminator output for fake images
    fake_output = discriminator(fake_images)
    # Discriminator output for real images
    real_output = discriminator(x)
    loss = -(torch.log(real_output) + (torch.log(1-fake_output))).mean()
    #print("d:", loss)
    return loss




loss_func = nn.BCELoss() # Sets the loss function for the initial discriminator training
epochs = 10 # Epochs for main training loop

def main_training_loop():
    """
        Main training loop for the GAN.

        This function trains both the discriminator and the generator over multiple epochs.

        Returns

        None
    """

    # Initial train discriminator
    initial_train_d(1, discriminator, loss_func, torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)))

    # Main loop
    for i in range(epochs):
        # Load a batch of real images
        for data in dataloader:
            try:
                real_images = data[0]
            except StopIteration:
                dataiter = iter(dataloader)
                real_images, labels = next(dataiter)

            batch_size = real_images.size(0)
            noise_vector = torch.randn(batch_size, noise_dim)

            # Train generator
            train_g(noise_vector, torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)))
            # Train discriminator
            # Add noise to real images to reduce mode collapse
            real_images += 0.05 * torch.randn_like(real_images)
            train_d(torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)), noise_vector, real_images)


main_training_loop()