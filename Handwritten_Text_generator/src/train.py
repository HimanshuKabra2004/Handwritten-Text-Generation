import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs=20, save_path='models/model.pth'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = inputs.view(outputs.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), save_path)
    print("Model saved to", save_path)