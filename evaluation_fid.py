import torch
from PIL import Image
import numpy as np
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import pandas as pd

# Assume images1 and images2 are two lists of PIL Image objects.
# Here's how you would define a function to calculate the FID score:

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
            img = img.resize((299, 299), Image.LANCZOS)  # Inception v3 expects 299x299 inputs
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

# Example usage:
inception_model = inception_v3_pool3_feature_extractor()
# images1 = [Image.open(path1), ...]  # List of PIL Images for dataset 1
# images2 = [Image.open(path2), ...]  # List of PIL Images for dataset 2
# fid_score = calculate_fid(images1, images2, inception_model)
# print("FID Score:", fid_score)
