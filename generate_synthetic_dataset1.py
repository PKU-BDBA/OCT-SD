
from diffusers import StableDiffusionPipeline, UNet2DConditionModel,StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from io import BytesIO
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from torchvision import models


from PIL import Image

transform = transforms.Compose([
    transforms.Resize([512,512]),
    # transforms.RandomResizedCrop(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 适应您的类别数
model.load_state_dict(torch.load('/home/luyx/OCT-SD/OCT-classification/quality-control/best_model.pth',map_location="cpu"))


def predict_image(image, model, transform):
    
    # 应用预处理
    image = transform(image)

    # 添加批次维度并转移到相应的设备
    image = image.unsqueeze(0)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        print(probabilities)
        # 使用0.4作为阈值
        predicted = (probabilities[0][1] > 0.35).float()
        return predicted.item()


model_path = "/home/luyx/OCT-SD/diffusers-main/examples/text_to_image/sd-OCT-model"

unet = UNet2DConditionModel.from_pretrained(model_path + "/checkpoint-4600/unet")

pipe = StableDiffusionPipeline.from_pretrained("/home/luyx/stable-diffusion-roentgen/Xray/roentgen_v1.0/roentgen",unet=unet,safety_checker=None).to("cuda:1")

import pandas as pd

train_df=pd.read_csv("/home/luyx/OCT-SD/OCT_dataset/train_OCT.csv")
train_df.loc[train_df["diagnosis"].str.contains("病变", na=False), "diagnosis"] = "DR"
train_df.loc[train_df["diagnosis"].str.contains("水肿", na=False), "diagnosis"] = "DME"
train_df.loc[train_df["diagnosis"].str.contains("玻璃膜", na=False), "diagnosis"] = "DRUSEN"
train_df.loc[train_df["diagnosis"].str.contains("黄斑变性", na=False), "diagnosis"] = "AMD"
train_df.loc[train_df["diagnosis"].str.contains("AMD", na=False), "diagnosis"] = "AMD"
train_df.loc[train_df["diagnosis"].str.contains("DME", na=False), "diagnosis"] = "DME"
train_df.loc[train_df["diagnosis"].str.contains("血管", na=False), "diagnosis"] = "PCV"
train_df.loc[train_df["diagnosis"].str.contains("PCV", na=False), "diagnosis"] = "PCV"
train_df.loc[train_df["diagnosis"].str.contains("正常", na=False), "diagnosis"] = "Normal"

extract_dataset=train_df[train_df["diagnosis"].isin(["DME","CNV","DR","DRUSEN","AMD","PCV"])]


# 创建以 'diagnosis' 为键，对应的 'text' 列表为值的字典
diagnosis_text_dict = {}
for _, row in extract_dataset.iterrows():
    diagnosis = row['diagnosis']
    text = row['text']
    if diagnosis in diagnosis_text_dict:
        if text not in diagnosis_text_dict[diagnosis]:
            diagnosis_text_dict[diagnosis].append(text)
    else:
        diagnosis_text_dict[diagnosis] = [text]

import pandas as pd

df=pd.read_csv("/home/luyx/OCT-SD/OCT_dataset/generate-1207/diagnosis_images.csv")[["Text","report"]]
text_report_dict = {row['Text']: row['report'] for index, row in df.iterrows()}
text_report_dict["there was no obvious abnormality in the shape of the macula"]="正常"
text_report_dict["normal"]="正常"

from tqdm import tqdm
import random
import os
import csv
import pandas as pd
from PIL import Image
import io

def save_image(image, path):
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

csv_file = '/home/luyx/OCT-SD/OCT_dataset/generate-1214/diagnosis_images.csv'
ground_truth_df = pd.read_csv("/home/luyx/OCT-SD/OCT_dataset/train_OCT.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Text', 'ImagePath'])

try:
    text_dict={}
    image_dict={}
    for key, values in diagnosis_text_dict.items():
        for value in values:
            if len(ground_truth_df[ground_truth_df["text"]==value]["image"].tolist())>10:
                text_dict[value]=len(ground_truth_df[ground_truth_df["text"]==value]["image"].tolist())
                ground_truth_figure=ground_truth_df[ground_truth_df["text"]==value].sample(10)
                # for i, image_path in enumerate(ground_truth_figure["image"].tolist()[:10]):
                #     image=Image.open(image_path)
                #     image_path = f"/home/luyx/OCT-SD/OCT_dataset/generate-1214/{text_report_dict[value]}/syn_{i}.jpg"
                #     os.makedirs(os.path.dirname(image_path), exist_ok=True)
                #     save_image(image, image_path)
                    
                # if os.path.exists(f"/home/luyx/OCT-SD/OCT_dataset/generate-1214/{text_report_dict[value]}"):
                #     continue
                
                images = pipe(prompt=value, num_images_per_prompt=50).images
                for i, image in enumerate(images):
                    image_path = f"/home/luyx/OCT-SD/OCT_dataset/generate-1214/{text_report_dict[value]}/syn_{i+11}.jpg"
                    while not predict_image(image,model,transform):
                        image=pipe(prompt=value, num_images_per_prompt=1).images[0]
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    save_image(image, image_path)
                    
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([value, image_path])
                
                # images = pipe(prompt=value, num_images_per_prompt=50).images
                # for i, image in enumerate(images):
                #     image_path = f"/home/luyx/OCT-SD/OCT_dataset/generate-1214/{text_report_dict[value]}/syn_{i+61}.jpg"
                #     while not predict_image(image,model,transform):
                #         image=pipe(prompt=value, num_images_per_prompt=1).images[0]
                #     os.makedirs(os.path.dirname(image_path), exist_ok=True)
                #     save_image(image, image_path)
                    
                #     with open(csv_file, mode='a', newline='') as file:
                #         writer = csv.writer(file)
                #         writer.writerow([value, image_path])
                        
except Exception as e:
    print(f"An error occurred: {e}")