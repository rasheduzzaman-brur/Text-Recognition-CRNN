import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import torch.nn as nn
from  loss import  CustomCTCLoss
from model import CRNN
from dataset import data_loader, test_data_loader
from utils import convert_labels_to_sequences, decode , calculate_accuracy, save_checkpoint,load_checkpoint
alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%="""
imgH = 32  # Height of the input image
nc = 1  # Number of input channels (grayscale)
nclass = len(alphabet) + 1  # Number of output classes
nh = 256  # Number of hidden units in the RNN

model = CRNN(32, 1, nclass, nh)
criterion = CustomCTCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
num_epochs = 50
# Parameters


# Step 5: Training loop
def train(model, data_loader, test_data_loader, criterion, optimizer, num_epochs, best_word_acc):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in data_loader:
            images = images.to(device)
            targets, lengths = convert_labels_to_sequences(labels)

            optimizer.zero_grad()
            output = model(images)
            output = output.log_softmax(2)

            input_lengths = torch.full((output.size(1),), output.size(0), dtype=torch.long)
            target_lengths = torch.tensor(lengths, dtype=torch.long)

            loss = criterion(output, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

        # Calculate accuracy on training data (or use a validation set)
        model.eval()
        decoded_texts = []
        true_texts = []
        with torch.no_grad():
            for images, labels in test_data_loader:
                images = images.to(device)
                output = model(images)
                output = output.log_softmax(2)
                decoded_texts.extend(decode(output))
                true_texts.extend(labels)

        char_acc, word_acc = calculate_accuracy(decoded_texts, true_texts)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Char Acc: {char_acc}, Word Acc: {word_acc}')

        # Save checkpoint if the current model has the best word accuracy
        if word_acc > best_word_acc:
            best_word_acc = word_acc
            save_checkpoint(model, optimizer, epoch, best_word_acc)

        model.train()
# Start training
best_word_acc = 0
train(model, data_loader, test_data_loader, criterion, optimizer, num_epochs, best_word_acc)
