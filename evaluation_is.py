import torch
from PIL import Image
import numpy as np
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

def inception_v3_pool3_feature_extractor():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

def calculate_is(images, inception):
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

    probs = get_inception_probabilities(images, inception)
    marginal_prob = np.mean(probs, axis=0)
    conditional_entropy = np.mean([entropy(prob) for prob in probs])
    marginal_entropy = entropy(marginal_prob)

    return np.exp(marginal_entropy - conditional_entropy)

# Example usage:
inception_model = inception_v3_pool3_feature_extractor()
# images = [Image.open(path1), ...]  # List of PIL Images for generated dataset
# is_score = calculate_is(images, inception_model)
# print("Inception Score:", is_score)
