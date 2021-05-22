import os
def get_mapping_dict(train_root = "./data/imagenette2/train"):
    mapping_folder_to_name = {}
    with open("./data/mapping.txt") as f:
        for line in f:
            line = line.strip().split(" ")
            mapping_folder_to_name[line[0]] = line[2]

    label = 0
    mapping_folder_to_label = {}
    mapping_name_to_label = {}
    mapping_label_to_name = {}

    for folder in sorted(os.listdir(train_root)):
        name = mapping_folder_to_name[folder]
        mapping_folder_to_label[folder] = label
        mapping_name_to_label[name] = label
        mapping_label_to_name[label] = name
        label += 1
    return mapping_folder_to_name, mapping_folder_to_label, mapping_name_to_label, mapping_label_to_name