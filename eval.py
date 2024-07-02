import torch.nn as nn
import torch
import torch.nn.functional as F
def decode_predictions(preds, alphabet):
    preds = preds.permute(1, 0, 2)  # Change to (batch, time, class)
    preds = F.log_softmax(preds, dim=2)
    preds = torch.argmax(preds, dim=2)
    preds = preds.cpu().numpy()

    decoded = []
    for pred in preds:
        decoded_label = ''
        for p in pred:
            if p != 0 and (len(decoded_label) == 0 or p != decoded_label[-1]):
                decoded_label += alphabet[p - 1]
        decoded.append(decoded_label)
    return decoded

def evaluate(model, data_loader, alphabet):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = decode_predictions(outputs, alphabet)
            correct += sum([p == l for p, l in zip(preds, labels)])
            total += len(labels)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

evaluate(crnn, data_loader, alphabet)
