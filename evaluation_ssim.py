import numpy as np
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from PIL import Image

def calculate_ms_ssim(image1, image2):
    def preprocess(img):
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((512, 512), Image.LANCZOS)  # Resize for consistency
        img = np.array(img).astype(np.float32)
        img /= 255  # Normalize
        return img

    image1 = preprocess(image1)
    image2 = preprocess(image2)
    scales = [1, 0.5, 0.25]  # Define scales for MS-SSIM
    msssim = []

    for scale in scales:
        # Resize images according to the current scale
        img1_resized = resize(image1, (int(image1.shape[0] * scale), int(image1.shape[1] * scale)), anti_aliasing=True)
        img2_resized = resize(image2, (int(image2.shape[0] * scale), int(image2.shape[1] * scale)), anti_aliasing=True)

        # Calculate SSIM for the current scale
        ssim_value = ssim(img1_resized, img2_resized, data_range=img1_resized.max() - img1_resized.min())
        msssim.append(ssim_value)

    # Calculate the product of SSIM values and take the nth root where n is the number of scales
    ms_ssim_value = np.prod(msssim) ** (1.0 / len(msssim))
    return ms_ssim_value

# Example usage:
# image1 = Image.open("path_to_image_1.jpg")
# image2 = Image.open("path_to_image_2.jpg")
# ms_ssim_score = calculate_ms_ssim(image1, image2)
# print("MS-SSIM Score:", ms_ssim_score)
