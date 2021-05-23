import os
from PIL import Image
import torchvision.transforms as transforms

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    data_root = "./data/imagenette2/val"
    for folder in os.listdir(data_root):
        os.mkdir(f"./data/imagenette2/val_small/{folder}/")
        for image_file in os.listdir(os.path.join(data_root, folder)):
            image_path = os.path.join(data_root, folder, image_file)
            re_save_path = f"./data/imagenette2/val_small/{folder}/{image_file}"
            img = Image.open(image_path).convert('RGB')
            img = transform(img)
            img.save(re_save_path)