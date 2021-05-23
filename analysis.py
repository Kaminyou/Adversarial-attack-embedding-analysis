from src.dataset import ImageNetteDataset
from src.models import get_embedding_model
from attack.utils import state_dict_key_transform
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def get_latent(embedding):
    latent_embedded = TSNE(n_components=2, init="pca").fit_transform(embedding)
    return latent_embedded

def draw_distribution(latent_embedded, label_all, predict_all, ori_or_adv, mapping_label_to_name, title=None):
    mapping_adv = {0:"false", 1:"true"}
    re_label_all = [mapping_label_to_name[i] for i in label_all]
    re_predict_all = [mapping_label_to_name[i] for i in predict_all]
    re_ori_or_adv = [mapping_adv[i] for i in ori_or_adv]
    df = pd.DataFrame({"dim_1":latent_embedded[:,0], "dim_2":latent_embedded[:,1], "labels":re_label_all, "predict":re_predict_all, "adv":re_ori_or_adv})
    hue_order = [mapping_label_to_name[i] for i in range(10)]
    
    plt.figure(figsize=(15,15))
    sns.scatterplot(data=df[df["adv"] == "false"], x="dim_1", y="dim_2", hue="labels", style="predict", palette="Accent", s=40, hue_order = hue_order, style_order=hue_order)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    title_sentence = "[Original data]"
    if title:
        title_sentence += title
    plt.title(title_sentence)
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,15))
    sns.scatterplot(data=df[df["adv"] == "true"], x="dim_1", y="dim_2", hue="labels", style="predict", palette="Accent", s=40, hue_order = hue_order, style_order=hue_order)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    title_sentence = "[Adversarial examples]"
    if title:
        title_sentence += title
    plt.title(title_sentence)
    plt.show()

def model_embedding(model, model_to, data, layer):
    if model_to == "densenet121":
        embedding = model.embedding(data).cpu()
    else:
        if layer == "first":
            embedding = model.embedding_first(data).cpu()
        elif layer == "layer1":
            embedding = model.embedding_layer1(data).cpu()
        elif layer == "layer2":
            embedding = model.embedding_layer2(data).cpu()
        elif layer == "layer3":
            embedding = model.embedding_layer3(data).cpu()
        elif layer == "layer4":
            embedding = model.embedding_layer4(data).cpu()
    return embedding

def embedding_pipeline(model_from, model_to, attack_mode, device, layer, mapping_folder_to_label):
    val_dataset = ImageNetteDataset(data_root="./data/imagenette2/val", mapping_folder_to_label=mapping_folder_to_label, train=True)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    adv_dataset = ImageNetteDataset(data_root=f"./adv_example/{attack_mode}/{model_from}/", mapping_folder_to_label=mapping_folder_to_label, train=True, simple_transform=True)
    adv_dataloader = DataLoader(adv_dataset, batch_size=100, shuffle=False)

    model = get_embedding_model(model_to)
    state_dict = state_dict_key_transform(f"./models/{model_to}.pth")
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    ori_or_adv = []
    label_all = []
    predict_all = []
    with torch.no_grad():
        for idx, (data, labels) in enumerate(val_dataloader):
            print(f"val process {idx + 1} / {len(val_dataloader)}             ", end = "\r")
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            embedding = model_embedding(model, model_to, data, layer)

            if idx == 0:
                embedding_all = embedding
            else:
                embedding_all = torch.vstack((embedding_all, embedding))

            _, preds = torch.max(output, 1)
            ori_or_adv += [0] * len(labels)
            label_all += list(labels.cpu().numpy())
            predict_all += list(preds.cpu().numpy())

        for idx, (data, labels) in enumerate(adv_dataloader):
            print(f"adv process {idx + 1} / {len(adv_dataloader)}             ", end = "\r")
            data = data.to(device)
            labels = labels.to(device)

            output = model(data)
            embedding = model_embedding(model, model_to, data, layer)

            embedding_all = torch.vstack((embedding_all, embedding))

            _, preds = torch.max(output, 1)
            ori_or_adv += [1] * len(labels)
            label_all += list(labels.cpu().numpy())
            predict_all += list(preds.cpu().numpy())

    embedding_all = embedding_all.numpy()
    return embedding_all, label_all, predict_all, ori_or_adv