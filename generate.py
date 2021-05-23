import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from src.utils import get_mapping_dict
from src.dataset import ImageNetteDataset
from src.models import get_model
from torch.utils.data import DataLoader

from attack.attacker import Attacker
from attack.deepfool import DeepFool
from attack.utils import single_image_process, reverse_transform, np_array_save_img, diff_calculate, int_to_torch_long, rgb_image_to_tensor
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # basic setting
    parser.add_argument('--attack_name', type=str, default="fgsm_un", help='attack mode:[fgsm_un, pgd_un, pgd_ta, mifgsm_un, deepfool]')
    parser.add_argument('--model_name', type=str, default="resnet18", help='model_name')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--val_root', type=str, default="./data/imagenette2/val/", help="validation root path")
    args = parser.parse_args()
    
    device = args.device
    model_name = args.model_name
    val_root = args.val_root
    attack_name = args.attack_name
    print(f"Generate {attack_name} for {model_name}")

    total_file_num = 0
    for folder in sorted(os.listdir(val_root)):
        for file in os.listdir(os.path.join(val_root, folder)):
            total_file_num += 1

    if not os.path.exists(f"./adv_example/{attack_name}/{model_name}"):
        os.makedirs(f"./adv_example/{attack_name}/{model_name}")

    mapping_folder_to_name, mapping_folder_to_label, mapping_name_to_label, mapping_label_to_name = get_mapping_dict()

    model = get_model(model_name)
    model.load_state_dict(torch.load(f"./models/{model_name}.pth"))

    model.to(device)

    criterion = nn.CrossEntropyLoss()


    file_num = 0
    success_num = 0
    error_num = 0
    for folder in sorted(os.listdir(val_root)):
        if not os.path.exists(f"./adv_example/{attack_name}/{model_name}/{folder}"):
            os.makedirs(f"./adv_example/{attack_name}/{model_name}/{folder}")
        for file in os.listdir(os.path.join(val_root, folder)):
            file_num += 1
            
            img_path = os.path.join(val_root, folder, file)
            label = mapping_folder_to_label[folder]
            img, img_tensor = single_image_process(img_path)
            img_tensor = img_tensor.to(device)
            success = False
            adv_pred_new = -1
            max_diff = -1
            with torch.no_grad():
                model.eval()
                output = model(img_tensor)
            pred = output.argmax().cpu().detach().numpy()
            
            # success classify => do adv generation
            if (label == pred):
                if attack_name != "deepfool":
                    attacker = Attacker(model = model, data = img_tensor, label=label, criterion = criterion, device=device, logging=False)
                    if attack_name == "fgsm_un":
                        adv, adv_pred = attacker.fgsm_untargeted_attack(epsilon=0.1)
                    elif attack_name == "pgd_un":
                        adv, adv_pred = attacker.pgd_untargeted_attack(eps=0.02, alpha=0.002, PGD_round=20)
                    elif attack_name == "pgd_ta":
                        adv, adv_pred = attacker.pgd_targeted_attack(eps=0.03, alpha=0.002, PGD_round=40, target=0)
                    elif attack_name == "mifgsm_un":
                        adv, adv_pred = attacker.mifgsm_untargeted_attack(eps=0.02, alpha=0.002, MIFGSM_round=20)
                else:
                    deepfool = DeepFool(model = model, steps=50, overshoot=0.02)
                    adv, adv_pred = deepfool(images=img_tensor, labels=int_to_torch_long(label), return_target_labels=True)
                
                adv = reverse_transform(adv)
                max_diff = diff_calculate(img, adv).max()
                if adv_pred != -1:
                    adv_tensor = rgb_image_to_tensor(adv).to(device)
                    with torch.no_grad():
                        model.eval()
                        output = model(adv_tensor)
                        adv_pred_new = output.argmax().cpu().detach().numpy()
                        if adv_pred_new != label:
                            success = True
                            success_num += 1
                            np_array_save_img(adv, f"./adv_example/{attack_name}/{model_name}/{folder}/{file}")
                            
            else:
                error_num += 1
            
            print(f"[{file_num+1}/{total_file_num}] {file} ori {label} adv_pred {adv_pred} adv_pred_rgb {adv_pred_new} max_diff {max_diff}")
            
    print(f"#file {file_num} || #error {error_num} || #success {success_num}")