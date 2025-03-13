import os 
from PIL import Image 
from torch.utils.data import Dataset 
from torchvision import transforms

class CarvanaDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path + "test_images/" + image for image in os.listdir(root_path + "test_images/")])
            self.masks = sorted([root_path + "test_masks/" + mask for mask in os.listdir(root_path + "test_masks/")])
        else:
            self.images = sorted([root_path + "train_images/" + image for image in os.listdir(root_path + "train_images/")])
            self.masks = sorted([root_path + "train_masks/" + mask for mask in os.listdir(root_path + "train_masks/")])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask

