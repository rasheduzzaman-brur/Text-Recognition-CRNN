import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

import pandas as pd

class SyntheticDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label



csv_file = '/home/rashed/office/medical_prescrip_v2/dftrain_medicine.csv'
root_dir = '/home/rashed/office/medical_prescrip_v2/output_dir_6_csv_image'

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Create dataset and dataloader
dataset = SyntheticDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

val_csv_file = '/home/rashed/office/medical_prescrip_v2/dftest_medicine.csv'
val_root_dir = '/home/rashed/office/medical_prescrip_v2/output_dir_6_csv_image'

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Create dataset and dataloader
dataset = SyntheticDataset(csv_file=val_csv_file, root_dir=val_root_dir, transform=transform)
test_data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
