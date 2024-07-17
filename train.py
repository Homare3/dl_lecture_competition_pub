import torch
#import hydra
#from omegaconf import DictConfig
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import ViTMAEForPreTraining, ViTMAEConfig
from torchvision import transforms
from PIL import Image
import os

from tqdm import tqdm

from src.utils import set_seed


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image






#@hydra.main(version_base=None, config_path="configs", config_name="base")
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(3407)
    num_epochs = 20

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageDataset(image_dir='./data/train', transform=transform)
    valid_dataset = ImageDataset(image_dir='./data/valid', transform=transform)
    #size = int(0.9*len(dataset))
    #train_dataset = Subset(dataset, indices[:size])
    #valid_dataset = Subset(dataset, indices[size:])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    #config = ViTMAEConfig()
    #model = ViTMAEForPreTraining(config)
    model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-5)


    avg_train_loss = 0
    avg_val_loss = 0

    for epoch in range(num_epochs):

        train_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            outputs = model(batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    
    torch.save(model.state_dict(), 'mae_vit_model_2.pth')


if __name__ == "__main__":
    train()
