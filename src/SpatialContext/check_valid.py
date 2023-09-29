import os
from tqdm import tqdm
from src.config import config

def validate_image(filepath):
    from PIL import Image, UnidentifiedImageError,ImageFile
    try: img = Image.open(filepath).convert('RGB'); img.load()
    except UnidentifiedImageError : return False
    except (IOError, OSError) : return False
    else: return True

if __name__ == "__main__" :
    ctg_fl = sorted(os.listdir(config.FILE_PATH))[1:]
    for dir_ in tqdm(ctg_fl):
        folder_path = os.path.join(config.FILE_PATH, dir_)
        files = os.listdir(folder_path)

        images = [os.path.join(folder_path, f) for f in files]
        for img in tqdm(images):
            valid = validate_image(img)
            if not valid:
                os.remove(img) # corrupted 된 이미지 제거
