from  utils import load_checkpoint, decode, calculate_accuracy
from model import CRNN
import torch
import torch.optim as optim
from dataset import SyntheticDataset, test_data_loader
alphabet = 'abcdefghijklmnopqrstuvwxyzA BCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-()/|_.'
imgH = 32  # Height of the input image
nc = 1  # Number of input channels (grayscale)
nclass = len(alphabet) + 1  # Number of output classes
nh = 256  # Number of hidden units in the RNN

model = CRNN(32, 1, nclass, nh)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
checpoint_path = "/home/rashed/own_project/custom_ocr_system/checkpoint/best_checkpoint_medicine.pth"
epoch, best_word_acc = load_checkpoint(model,optimizer,checpoint_path)
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
print(f'Evaluation - Char Acc: {char_acc}, Word Acc: {word_acc}')