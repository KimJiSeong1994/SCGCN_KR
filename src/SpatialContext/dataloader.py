
class Get_data :
    def __init__(self, FILE_PATH) :
        self.FILE_PATH = FILE_PATH

    def train_test_split(self, split_ratio = 0.2) :
        import os
        train_images, test_images = [], []

        for ctg in sorted(os.listdir(self.FILE_PATH))[1:] :
            files = os.listdir(f'{FILE_PATH}/{ctg}')
            sorted(files)[1:].sort(key = lambda f: int(''.join(filter(str.isdigit, f))))





