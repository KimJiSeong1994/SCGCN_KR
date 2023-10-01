
class Get_data :
    def __init__(self, FILE_PATH) :
        self.FILE_PATH = FILE_PATH

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

        random.shuffle(train_images)
        random.shuffle(test_images)
        class_to_idx = {os.path.basename(f): idx for idx, f in enumerate(sorted(os.listdir(FILE_PATH))[1:])}
        return class_to_idx, train_images, test_images





