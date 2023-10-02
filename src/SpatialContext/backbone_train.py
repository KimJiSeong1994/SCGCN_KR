
if __name__ == "__main__" :
    import numpy as np
    import pandas as pd
    from src.config import config
    from src.SpatialContext.backbone_utils import CustomImageDataset, train_test_split
    from src.SpatialContext.trainner import model_train, model_evaluate

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

    num_epochs = 50
    min_loss = np.inf
    for epoch in range(num_epochs):
        train_loss, train_acc = model_train(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = model_evaluate(model, test_loader, loss_fn, device)
        if val_loss < min_loss:
            print(f'[INFO] val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. Saving Model!')
            min_loss = val_loss
            torch.save(model.state_dict(), './src/SpatialContext/model_output/backbone_result.pth')
        print(f'epoch {epoch + 1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f}')

    model.load_state_dict(torch.load('./src/SpatialContext/model_output/backbone_result.pth'))
    final_loss, final_acc = model_evaluate(model, test_loader, loss_fn, device)
    print(f'evaluation loss: {final_loss:.5f}, evaluation accuracy: {final_acc:.5f}') # eval
