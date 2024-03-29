import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import os
import csv


# model = YourCustomModel() # Replace with your model loading code
# pipe = YourImageGenerationPipeline() # Replace with your pipeline setup code

transform = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
])

def predict_image(image, model, transform):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities[0][1] > 0.35).float()
        return predicted.item()

def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

# Example of a process to generate and classify images, then save those meeting a condition
def generate_and_classify_images(prompt_list, pipe, model, transform, output_dir, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Text', 'ImagePath'])

    for prompt in tqdm(prompt_list):
        try:
            images = pipe(prompt=prompt, num_images_per_prompt=10).images  # Generate images
            for i, image in enumerate(images):
                if predict_image(image, model, transform):  # Classify image
                    image_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}_{i}.jpg")
                    save_image(image, image_path)
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([prompt, image_path])
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {e}")

# Example usage:
# prompt_list = ['A sample prompt 1', 'Another example prompt 2']  # Define your prompts
# output_dir = 'path/to/output/dir'
# csv_file_path = 'path/to/output/csv_file.csv'
# generate_and_classify_images(prompt_list, pipe, model, transform, output_dir, csv_file_path)
