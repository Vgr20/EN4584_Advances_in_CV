import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

from carvana_dataset import CarvanaDataset
from unet import UNet

if __name__== "__main__":
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    DATA_FOLDER = "data/"
    MODEL_CHECKPOINT = "models/unet_carvana.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = CarvanaDataset(DATA_FOLDER)
    generator =torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.8,0.2], generator=generator)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, num_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        train_running_loss = 0.0
        for idx, (images, masks) in enumerate(train_loader):
            images = images.float().to(DEVICE)
            masks = masks.float().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for idx, (images, masks) in enumerate(val_loader):
                images = images.float().to(DEVICE)
                masks = masks.float().to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()

            val_loss = val_running_loss / len(val_loader)

        print("-" * 30)
        print(f"Epoch: {epoch+1} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        print("-" * 30)

    torch.save(model.state_dict(), MODEL_CHECKPOINT)


