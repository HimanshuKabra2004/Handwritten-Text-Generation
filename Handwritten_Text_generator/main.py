import argparse
import torch
from src.data_loader import IAMDataset
from src.model import HandwritingGenerator
from src.train import train_model
from src.generate import generate_handwriting
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def main():
    parser = argparse.ArgumentParser(description="Handwritten Text Generator")
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='generate',
                        help="Choose 'train' to train the model or 'generate' to generate handwriting.")
    parser.add_argument('--text', type=str, default='Hello world!',
                        help="Text to generate handwriting for.")
    parser.add_argument('--style', type=str, default=None,
                        help="Optional style vector or reference for handwriting.")
    parser.add_argument('--checkpoint', type=str, default='models/model.pth',
                        help="Path to save/load the model.")
    args = parser.parse_args()

    if args.mode == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = IAMDataset(root_dir='data/iam', transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        model = HandwritingGenerator()
        train_model(model, dataloader, epochs=20, save_path=args.checkpoint)

    elif args.mode == 'generate':
        model = HandwritingGenerator()
        if os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint))
            model.eval()
            generate_handwriting(model, args.text, style=args.style)
        else:
            print(f"❌ Checkpoint not found at {args.checkpoint}. Train the model first.")

if __name__ == '__main__':
    main()
