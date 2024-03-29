import torch
from PIL import Image
import numpy as np
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import pandas as pd
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

fid_list=[]
for epoch in range(2200,4700,100):

    unet_model_path = f"/home/luyx/OCT-SD/diffusers-main/examples/text_to_image/sd-OCT-model/checkpoint-{epoch}/unet"
    pipeline_model_path = "/home/luyx/stable-diffusion-roentgen/Xray/roentgen_v1.0/roentgen"

    try:
        unet = UNet2DConditionModel.from_pretrained(unet_model_path)
        pipe = StableDiffusionPipeline.from_pretrained(pipeline_model_path, unet=unet,safety_checker=None)
        pipe.to("cuda")
    except Exception as e:
        print(f"Model loading error: {e}")
        exit()

    try:
        train_df = pd.read_csv("/home/luyx/OCT-SD/OCT_dataset/train_OCT.csv")
    except FileNotFoundError:
        print("CSV file not found.")
        exit()

    def inception_v3_pool3_feature_extractor():
        model = inception_v3(pretrained=True, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        return model

    def calculate_fid(images1, images2, inception):
        def get_inception_features(images, inception):
            features = []
            for img in images:
                img = img.convert('RGB')
                img = img.resize((512, 512), Image.LANCZOS)
                img = np.array(img).astype(np.float32)
                img /= 255
                img = img.transpose((2, 0, 1))
                img = torch.tensor(img).unsqueeze(0)
                with torch.no_grad():
                    pred = inception(img)[0]
                features.append(pred.squeeze().numpy())
            return np.array(features)

        feat1 = get_inception_features(images1, inception)
        feat2 = get_inception_features(images2, inception)

        mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
        offset = np.eye(sigma1.shape[0]) * 1e-6

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        covmean = sqrtm((sigma1+offset).dot(sigma2+offset))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    inception_model = inception_v3_pool3_feature_extractor()
    prompt_list = train_df[train_df.groupby('text')['text'].transform('count') > 10]["text"].drop_duplicates().tolist() # NOTE Change

    all_fid_score = []
    for p in prompt_list:
        try:
            generated_images = pipe(prompt=p, num_images_per_prompt=4).images # NOTE Change
            sample_image_paths = train_df[train_df["text"] == p]["image"].sample(4).tolist() # NOTE Change
            sample_images = [Image.open(path) for path in sample_image_paths]
            fid_score = calculate_fid(generated_images, sample_images, inception_model)
            all_fid_score.append(fid_score)
        except Exception as e:
            print(f"Error processing prompt '{p}': {e}")

    if all_fid_score:
        average_fid_score = sum(all_fid_score) / len(all_fid_score)
        print(f"Epoch {epoch}: Average FID Score:", average_fid_score)
        fid_list.append(average_fid_score)
    else:
        print("No FID scores were calculated.")

import numpy as np
import matplotlib.pyplot as plt

x_values = np.arange(2200, 4700, 100)[:len(fid_list)]

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(x_values, fid_list, marker='o', linestyle='-', color='b')
plt.title("Line Graph of Given Data Points")
plt.xlabel("X-axis (2200-4700, 100 intervals)")
plt.ylabel("Value")
plt.grid(True)

plt.savefig('fid.png')


with open("fid.txt", "w") as f:
    for i in fid_list:
        f.write(str(i) + '\n')