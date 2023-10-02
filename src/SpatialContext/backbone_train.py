def model_train(model, data_loader, loss_fn, optimizer, device):
    from tqdm import tqdm
    model.train()
    running_loss, corr= 0, 0
    prograss_bar = tqdm(data_loader)

    for img, lbl in prograss_bar:
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = loss_fn(output, lbl)
        loss.backward()
        optimizer.step()
        _, pred = output.max(dim=1)
        corr += pred.eq(lbl).sum().item()
        running_loss += loss.item() * img.size(0)

    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc


def model_evaluate(model, data_loader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        corr, running_loss = 0, 0

        for img, lbl in data_loader:
            img, lbl = img.to(device), lbl.to(device)
            output = model(img)
            _, pred = output.max(dim=1)
            corr += torch.sum(pred.eq(lbl)).item()
            running_loss += loss_fn(output, lbl).item() * img.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

if __name__ == "__main__" :
    import pandas as pd
    from src.config import config
    from src.SpatialContext.backbone_utils import CustomImageDataset

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models
    from torchvision import transforms
    from torch.utils.data import DataLoader

    FILE_PATH = config.FILE_PATH
    seoul_info = pd.read_pickle("./data/landuses/seoul_landuse.pkl")
    seoul_info['index'] = seoul_info.reset_index()["index"].astype(str)
    train_images, test_images, train_labels, test_labels, class_to_idx = train_test_split(DATA = seoul_info, FILE_PATH = FILE_PATH, split_ratio = 0.2)

    func_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
    train_dataset = CustomImageDataset(files = train_images, labels = train_labels, class_to_idx = class_to_idx, transform = func_transform)
    test_dataset = CustomImageDataset(files = test_images, labels = test_labels, class_to_idx = class_to_idx, transform = func_transform)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = models.resnet101(pretrained = True) # resent backbone
    for param in model.parameters(): param.requires_grad = False # weigth freeze
    model.fc = nn.Sequential(nn.Linear(2048, 200), nn.ReLU(), nn.Linear(200, 5)) # model custom
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = nn.CrossEntropyLoss()




