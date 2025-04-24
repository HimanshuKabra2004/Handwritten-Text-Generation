import torch
import matplotlib.pyplot as plt
from src.model import HandwritingGenerator

def generate_handwriting(model, text, style=None):
    # Dummy generator: returns a blank image with input text overlay
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('L', (512, 128), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((10, 50), text, fill=0, font=font)
    plt.imshow(img, cmap='gray')
    plt.title('Generated Handwriting')
    plt.axis('off')
    plt.show()


def generate_text(text):
    model = HandwritingGenerator()
    model.eval()
    print(f"Generating handwriting for: {text}")
    # Mock-up output
    return f"[Handwritten Image of: '{text}']"