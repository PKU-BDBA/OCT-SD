import numpy as np
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from PIL import Image
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


ssim_list=[]
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

    def calculate_ms_ssim(image1, image2):
        def preprocess(img):
            img = img.convert('L')
            img = img.resize((512, 512), Image.LANCZOS)
            img = np.array(img).astype(np.float32)
            img /= 255
            return img
        image1=preprocess(image1)
        image2=preprocess(image2)
        scales = [1, 0.5, 0.25]
        msssim = []

        for scale in scales:
            img1_resized = resize(image1, (int(image1.shape[0] * scale), int(image1.shape[1] * scale)), anti_aliasing=True)
            img2_resized = resize(image2, (int(image2.shape[0] * scale), int(image2.shape[1] * scale)), anti_aliasing=True)

            ssim_value = ssim(img1_resized, img2_resized, data_range=img1_resized.max() - img1_resized.min(), multichannel=True)
            msssim.append(ssim_value)

        ms_ssim_value = np.prod(msssim) ** (1.0 / len(msssim))
        return ms_ssim_value

    prompt_list = train_df[train_df.groupby('text')['text'].transform('count') > 10]["text"].drop_duplicates().tolist() # NOTE Change
    all_ms_ssim_scores = []
    for prompt in prompt_list:
        try:
            generated_images = pipe(prompt=prompt, num_images_per_prompt=4).images # NOTE Change
            total_score = []

            for i in range(len(generated_images)):
                for j in range(i + 1, len(generated_images)):
                    total_score.append(calculate_ms_ssim(generated_images[i], generated_images[j]))

            all_ms_ssim_scores.append(sum(total_score)/len(total_score))
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")

    if all_ms_ssim_scores:
        average_ms_ssim_score = sum(all_ms_ssim_scores) / len(all_ms_ssim_scores)
        print(f"Epoch {epoch}: Average MS-SSIM Score:", average_ms_ssim_score)
        ssim_list.append(average_ms_ssim_score)
    else:
        print("No MS-SSIM scores were calculated.")


import numpy as np
import matplotlib.pyplot as plt

x_values = np.arange(2200, 4700, 100)[:len(ssim_list)]

# Plotting the line graph
plt.figure(figsize=(10, 6))
plt.plot(x_values, ssim_list, marker='o', linestyle='-', color='b')
plt.title("Line Graph of Given Data Points")
plt.xlabel("X-axis (2200-4700, 100 intervals)")
plt.ylabel("Value")
plt.grid(True)
plt.savefig('ssim.png')


with open("ssim.txt", "w") as f:
    for i in ssim_list:
        f.write(str(i) + '\n')
