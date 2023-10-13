
if __name__ == '__main__' :
    import pandas as pd
    from src.utils import Utils

    import torch
    import torch.nn as nn
    from torchvision import models

    train_df = pd.read_pickle('./data/landuses/seoul_landuse.pkl')
    PPE_ajd = Utils.Delaunay([[p.x, p.y] for p in train_df['geometry'].centroid])

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = models.resnet101(pretrained = True).to(device)  # ResNet backbone
    model.fc = nn.Sequential(nn.Linear(2048, 200), nn.ReLU(), nn.Linear(200, 5))  # model custom
    model.load_state_dict(torch.load('./src/SpatialContext/model_output/backbone_result.pth', map_location = torch.device('cpu')))

