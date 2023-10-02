from torch.utils.data import Dataset
class CustomImageDataset(Dataset):
    def __init__(self, files, labels, class_to_idx, transform):
        super(CustomImageDataset, self).__init__()
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image

        file = self.files[idx]
        img = Image.open(file).convert('RGB')
        img = self.transform(img)
        lbl = self.class_to_idx[self.labels[idx]]
        return img, lbl

def train_test_split(FILE_PATH, DATA, split_ratio = 0.2) :
    import os
    import random

    train_images, test_images = [], []
    for ctg in sorted(os.listdir(FILE_PATH))[1:] :
        files = sorted(os.listdir(f'{FILE_PATH}/{ctg}'))[1:]
        files = [f'{FILE_PATH}/{ctg}/' + f for f in files]
        sorted(files)[1:].sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

        random.seed(42)
        random.shuffle(files)
        idx = int(len(files) * split_ratio)
        train, test = files[:-idx], files[-idx:]
        train_images.extend(train)
        test_images.extend(test)

    random.shuffle(train_images)
    random.shuffle(test_images)
    class_to_idx = {os.path.basename(f): idx for idx, f in enumerate(sorted(os.listdir(FILE_PATH))[1:])}

    train_labels = DATA.loc[DATA["index"].isin([f.split('_')[-1].split(".")[0] for f in train_images]), 'landuse'].values.tolist()
    test_labels = DATA.loc[DATA["index"].isin([f.split('_')[-1].split(".")[0] for f in train_images]), 'landuse'].values.tolist()
    return train_images, test_images, train_labels, test_labels, class_to_idx

