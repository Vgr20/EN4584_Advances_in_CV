import torch 
import matplotlib.pyplot as plt 
from torchvision import transforms 
from PIL import Image 

from unet import UNet
from carvana_dataset import CarvanaDataset 

def pred_show_image_grid (data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    image_dataset = CarvanaDataset(data_path, test=True)
    images = []
    ground_truth_masks = []
    predicted_masks = []

    for img, ground_truth_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)
        with torch.no_grad():
            pred_mask = model(img.to(device))
        
        img = img.squeeze(0).cpu().detach().permute(1,2,0)

        pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1,2,0)
        # pred_mask = torch.sigmoid(pred_mask)
        pred_mask[pred_mask > 0] = 1
        pred_mask[pred_mask < 0] = 0

        ground_truth_mask = ground_truth_mask.squeeze(0).cpu().detach().permute(1,2,0)

        images.append(img)
        ground_truth_masks.append(ground_truth_mask)
        predicted_masks.append(pred_mask)

    images.extend(ground_truth_masks)
    images.extend(predicted_masks)

    fig = plt.figure(figsize=(15,15))
    for i in range(1, 3*len(image_dataset)+1):
        fig.add_subplot(len(image_dataset), 3, i)
        plt.imshow(images[i-1])
    plt.show()
    # Save the image 
    fig.savefig("saved_predictions/output.png")

def single_image_inference(image_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).float().to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        pred_mask = model(image.to(device))

    image = image.squeeze(0).cpu().detach().permute(1,2,0)
    pred_mask = pred_mask.squeeze(0).cpu().detach().permute(1,2,0)

    pred_mask[pred_mask > 0] = 1 
    pred_mask[pred_mask < 0] = 0

    fig = plt.figure(figsize=(15,15))
    for i in range(1, 3):
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(image)
        else:
            plt.imshow(pred_mask)
    plt.show()
    # Save the images 
    fig.savefig("saved_predictions/single_image_output.png")

if __name__ == "__main__":
    data_path = "data/"
    model_pth = "models/unet_carvana.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # pred_show_image_grid(data_path, model_pth, device)
    single_image_inference("data/test/0cdf5b5d0ce1_01.jpg", model_pth, device)
