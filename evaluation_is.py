import torch
from PIL import Image
import numpy as np
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import pandas as pd
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from scipy.stats import entropy

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

    def calculate_is(generated_images, inception):
        def get_inception_probabilities(images, inception):
            probs = []
            for img in images:
                img = img.convert('RGB')
                img = img.resize((299, 299), Image.LANCZOS)
                img = np.array(img).astype(np.float32)
                img /= 255
                img = img.transpose((2, 0, 1))
                img = torch.tensor(img).unsqueeze(0)
                with torch.no_grad():
                    pred = inception(img)[0]
                probs.append(torch.nn.functional.softmax(pred, dim=0).numpy())
            return np.array(probs)

        probs = get_inception_probabilities(generated_images, inception)
        marginal_prob = np.mean(probs, axis=0)
        conditional_entropy = np.mean([entropy(prob) for prob in probs])
        marginal_entropy = entropy(marginal_prob)

        return np.exp(marginal_entropy - conditional_entropy)


    inception_model = inception_v3_pool3_feature_extractor()
    prompt_list = train_df[train_df.groupby('text')['text'].transform('count') > 10]["text"].drop_duplicates().tolist() # NOTE Change

    all_fid_score = []
    for p in prompt_list:
        try:
            generated_images = pipe(prompt=p, num_images_per_prompt=4).images # NOTE Change
            sample_image_paths = train_df[train_df["text"] == p]["image"].sample(4).tolist() # NOTE Change
            sample_images = [Image.open(path) for path in sample_image_paths]
            fid_score = calculate_is(generated_images, inception_model)
            all_fid_score.append(fid_score)
        except Exception as e:
            print(f"Error processing prompt '{p}': {e}")

    if all_fid_score:
        average_fid_score = sum(all_fid_score) / len(all_fid_score)
        print(f"Epoch {epoch}: Average IS Score:", average_fid_score)
        fid_list.append(average_fid_score)
    else:
        print("No IS scores were calculated.")

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


plt.savefig('is.png')


with open("is.txt", "w") as f:
    for i in fid_list:
        f.write(str(i) + '\n')