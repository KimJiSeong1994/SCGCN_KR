class Get_data :
    def __init__(self, FILE_PATH, DATA) :
        self.FILE_PATH = FILE_PATH
        self.DATS = DATA
        self.DATA["index"] = self.DATA.reset_index()["index"].astype(str) # type-casting

    def train_test_split(self, split_ratio = 0.2) :
        import os
        import random

        train_images, test_images = [], []
        for ctg in sorted(os.listdir(self.FILE_PATH))[1:] :
            files = os.listdir(f'{self.FILE_PATH}/{ctg}')
            sorted(files)[1:].sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

            random.seed(42)
            random.shuffle(files)
            idx = int(len(files) * split_ratio)
            train, test = files[:-idx], files[-idx:]
            train_images.extend(train)
            test_images.extend(test)

        random.shuffle(self.train_images)
        random.shuffle(self.test_images)
        class_to_idx = {os.path.basename(f): idx for idx, f in enumerate(sorted(os.listdir(self.FILE_PATH))[1:])}

        train_labels = self.DATA.loc[self.DATA["index"].isin([f.split('_')[-1].split(".")[0] for f in train_images]), 'landuse'].values.tolist()
        test_labels = self.DATA.loc[self.DATA["index"].isin([f.split('_')[-1].split(".")[0] for f in train_images]), 'landuse'].values.tolist()
        return train_images, test_images, train_labels, test_labels, class_to_idx