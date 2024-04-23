# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np

# """ Convolutional block:
#     It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
# """
# class conv_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_c)

#         self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_c)

#         self.relu = nn.ReLU()

#     def forward(self, inputs):
#         x = self.conv1(inputs)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)

#         return x

# """ Encoder block:
#     It consists of an conv_block followed by a max pooling.
#     Here the number of filters doubles and the height and width half after every block.
# """
# class encoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.conv = conv_block(in_c, out_c)
#         self.pool = nn.MaxPool2d((2, 2))

#     def forward(self, inputs):
#         x = self.conv(inputs)
#         p = self.pool(x)

#         return x, p

# """ Decoder block:
#     The decoder block begins with a transpose convolution, followed by a concatenation with the skip
#     connection from the encoder block. Next comes the conv_block.
#     Here the number filters decreases by half and the height and width doubles.
# """
# class decoder_block(nn.Module):
#     def __init__(self, in_c, out_c):
#         super().__init__()

#         self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
#         self.conv = conv_block(out_c+out_c, out_c)

#     def forward(self, inputs, skip):
#         x = self.up(inputs)
#         x = torch.cat([x, skip], axis=1)
#         x = self.conv(x)

#         return x


# class build_unet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         """ Encoder """
#         self.e1 = encoder_block(3, 64)
#         self.e2 = encoder_block(64, 128)
#         self.e3 = encoder_block(128, 256)
#         self.e4 = encoder_block(256, 512)

#         """ Bottleneck """
#         self.b = conv_block(512, 1024)

#         """ Decoder """
#         self.d1 = decoder_block(1024, 512)
#         self.d2 = decoder_block(512, 256)
#         self.d3 = decoder_block(256, 128)
#         self.d4 = decoder_block(128, 64)

#         """ Classifier """
#         self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

#     def forward(self, inputs):
#         """ Encoder """
#         s1, p1 = self.e1(inputs)
#         s2, p2 = self.e2(p1)
#         s3, p3 = self.e3(p2)
#         s4, p4 = self.e4(p3)

#         """ Bottleneck """
#         b = self.b(p4)

#         """ Decoder """
#         d1 = self.d1(b, s4)
#         d2 = self.d2(d1, s3)
#         d3 = self.d3(d2, s2)
#         d4 = self.d4(d3, s1)

#         """ Classifier """
#         outputs = self.outputs(d4)

#         return outputs
# if __name__ == "__main__":
#     # Generate random input images
#     inputs = torch.randn((2, 3, 512, 512))
    
#     # Forward pass through the model
#     model = build_unet()
#     outputs = model(inputs)
    
#     # Convert outputs to probabilities (assuming it's a binary segmentation)
#     probs = torch.sigmoid(outputs)
    
#     # Convert tensors to numpy arrays for visualization
#     input_images = inputs.cpu().detach().numpy()
#     output_masks = probs.cpu().detach().numpy()

#     # Plot the random input images and their corresponding output masks
#     num_images = input_images.shape[0]
#     fig, axes = plt.subplots(num_images, 2, figsize=(8, 8*num_images))
#     for i in range(num_images):
#         axes[i, 0].imshow(np.transpose(input_images[i], (1, 2, 0)))
#         axes[i, 0].set_title("Input Image")
#         axes[i, 0].axis('off')
        
#         axes[i, 1].imshow(output_masks[i, 0], cmap='gray')
#         axes[i, 1].set_title("Output Mask")
#         axes[i, 1].axis('off')

#     plt.tight_layout()
#     plt.show()
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize,ToTensor,Compose
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
        outputs = self.outputs(d4)

        return outputs

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images_dir = os.path.join(data_dir, 'images')  # Directory containing images
        self.masks_dir = os.path.join(data_dir, 'masks')    # Directory containing masks
        self.image_filenames = os.listdir(self.images_dir)  # List of image filenames
        self.mask_filenames = os.listdir(self.masks_dir)    # List of mask filenames

    def __len__(self):
        return len(self.image_filenames)  # Assuming number of images equals number of masks

    def __getitem__(self, idx):
        # Load image and mask based on index
        img_name = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.masks_dir, self.mask_filenames[idx])
        
        # Open image and mask using PIL
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')  # Convert to grayscale mask

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Define transformations to resize images and convert them to tensors
transform = Compose([
    Resize((572, 572)),
    ToTensor()
])

if __name__ == "__main__":
    # Define dataset and dataloader
    data_dir = 'train/'
    # transform = ToTensor()  # Example transformation
    transform = transforms.Compose([
        transforms.Resize((572, 572)),
        transforms.ToTensor()
    ])

    resize_transform = Resize((572, 572))
    dataset = CustomDataset(data_dir, transform=resize_transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Create model instance
    model = build_unet()

    # Iterate over batches
    for images, masks in dataloader:
        # Forward pass
        outputs = model(images)
        print(outputs.shape)  # Shape of the output

        # Assuming binary segmentation, convert to probabilities
        probs = torch.sigmoid(outputs)

        # Visualize batch
        for i in range(images.size(0)):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(images[i].permute(1, 2, 0))  # Convert CHW to HWC
            plt.title('Input Image')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(masks[i].squeeze(), cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(probs[i].squeeze().detach().numpy(), cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.show()
