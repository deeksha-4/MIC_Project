import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.nn.functional import relu
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import pad
from torchvision.datasets import ImageFolder
import os
from PIL import Image

class Unet(nn.Module):
    def __init__(self, no_classes): # no of classes is for the segmentation task
        super().__init__()
        # inherits from nn.Module, initializes based on super class

        # encoder layer
        # each layer- 2 conv layesr with relu + max pooling layer
        # last layer - no max pooling

        self.l1c1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        # using padding 1 instead of 0(mentioned in paper) to avoid post processing
        self.l1c2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.l1p = nn.MaxPool2d(kernel_size=2, stride=2)

        # input 284x284x64
        self.l2c1 = nn.Conv2d(64, 128, kernel_size=3, padding=0) # output: 282x282x128
        self.l2c2 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # output: 280x280x128
        self.l2p = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.l3c1 = nn.Conv2d(128, 256, kernel_size=3, padding=0) # output: 138x138x256
        self.l3c2 = nn.Conv2d(256, 256, kernel_size=3, padding=0) # output: 136x136x256
        self.l3p = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.l4c1 = nn.Conv2d(256, 512, kernel_size=3, padding=0) # output: 66x66x512
        self.l4c2 = nn.Conv2d(512, 512, kernel_size=3, padding=0) # output: 64x64x512
        self.l4p = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.l5c1 = nn.Conv2d(512, 1024, kernel_size=3, padding=0) # output: 30x30x1024
        self.l5c2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=0) # output: 28x28x1024
        # no pooling cos last layer

        # decoder - upsampling + 2 conv layers
        # one final 1x1 conv for final outptu
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=0)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=0)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=0)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=0)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=0)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=0)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=0)

        # Output layer, 1x1 conv to map image to seg mask
        self.outconv = nn.Conv2d(64, no_classes, kernel_size=1)


    def forward(self, x):
        # encoder
        # x_perm = x.permute(2, 3, 1, 0).squeeze()

        # print(x.size())
        xe11 = relu(self.l1c1(x))
        # print("xe11", xe11.size())
        xe12 = relu(self.l1c2(xe11))
        # print("xe12", xe12.size())

        xp1 = self.l1p(xe12)

        # print("xp1", xp1.size())

        xe21 = relu(self.l2c1(xp1))
        xe22 = relu(self.l2c2(xe21))
        xp2 = self.l2p(xe22)

        # print("xp2", xp2.size())

        xe31 = relu(self.l3c1(xp2))
        xe32 = relu(self.l3c2(xe31))
        xp3 = self.l3p(xe32)

        # print("xp3", xp3.size())

        xe41 = relu(self.l4c1(xp3))
        xe42 = relu(self.l4c2(xe41))
        xp4 = self.l4p(xe42)

        # print("xp4", xp4.size())

        xe51 = relu(self.l5c1(xp4))
        xe52 = relu(self.l5c2(xe51))
        
        # Decoder
        # print(xe52.size())
        xu1 = self.upconv1(xe52)
        print(xu1.size(), xe42.size())
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    


class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images_dir = os.path.join(data_dir, 'images')  # Directory containing images
        self.masks_dir = os.path.join(data_dir, 'masks')    # Directory containing masks
        self.image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.JPG') or f.endswith('.jpg')])  # List of image filenames
        self.mask_filenames = sorted([f for f in os.listdir(self.masks_dir) if f.endswith('.PNG') or f.endswith('.png')])    # List of mask filenames

    def __len__(self):
        return len(self.image_filenames)  # Assuming number of images equals number of masks

    def __getitem__(self, idx):
        # Load image and mask based on index
        img_name = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.masks_dir, self.mask_filenames[idx])
        
        # Open image and mask using PIL
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')  # Convert to grayscale mask

        # Pad the image to achieve 572x572 size
        padding = (572 - image.size[0], 572 - image.size[1], 0, 0)  # Calculate padding values
        image = pad(image, padding)  # Pad with calculated values
        
        if self.transform:
            # Apply transformations if provided
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convert PIL image to tensor and transpose dimensions to [572, 572, 3]
        image = image.permute(1, 2, 0)
        
        assert image.size() == (572, 572, 3)  # Ensure the size matches
        
        return image, mask



if __name__ == '__main__':
    # transform = ToTensor()  # Example transformation
    transform = transforms.Compose([
    transforms.Resize((572, 572)),  # Resize to 572x572
    transforms.ToTensor()  # Convert PIL image to tensor
])

    no_classes = 2
    num_epochs = 5
    batch_size = 1
    learning_rate = 1e-5
    data_dir = ''
    # Instantiate your dataset classes for training and testing
    # train_dataset = CustomDataset("../dataset/OTU_2D/train", transform=transform)
    # test_dataset = CustomDataset("../dataset/OTU_2D/test", transform=transform)
    # train_dataset = CustomDataset("train/", transform=transform)
    # test_dataset = CustomDataset("test/", transform=transform)

    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)


    # Create data loaders for training and testing data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate your model
    model = Unet(no_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            # print(images.size())
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Testing loop
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            outputs = model(images)
            # Evaluate your model, compute metrics, etc.

