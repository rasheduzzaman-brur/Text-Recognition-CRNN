import torch
alphabet = """Only thewigsofrcvdampbkuq.$A-210xT5'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%="""
# Create a mapping from characters to integers
char_to_int = {char: i for i, char in enumerate(alphabet, 1)}  # Start indexing from 1

def convert_labels_to_sequences(labels):
    sequences = []
    lengths = []
    for label in labels:
        sequence = [char_to_int[char] for char in label]
        sequences.extend(sequence)
        lengths.append(len(sequence))
    return torch.tensor(sequences, dtype=torch.long), lengths

# Step 4: Helper functions for accuracy calculation
def calculate_accuracy(decoded_texts, true_texts):
    character_correct = 0
    character_total = 0
    word_correct = 0

    for decoded, true in zip(decoded_texts, true_texts):
        character_correct += sum(dc == tc for dc, tc in zip(decoded, true))
        character_total += len(true)
        if decoded == true:
            word_correct += 1

    character_accuracy = character_correct / character_total
    word_accuracy = word_correct / len(true_texts)
    return character_accuracy, word_accuracy

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, best_word_acc, filepath='/home/rashed/own_project/custom_ocr_system/checkpoint/best_checkpoint_medicine.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_word_acc': best_word_acc
    }
    torch.save(checkpoint, filepath)

# Function to load checkpoint
def load_checkpoint(model,optimizer,filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_word_acc = checkpoint['best_word_acc']
    return epoch, best_word_acc

def decode(output):
    output = output.permute(1, 0, 2)  # [b, w, c]
    output = torch.argmax(output, dim=2)  # [b, w]
    output = output.cpu().numpy()

    decoded_texts = []
    for seq in output:
        text = []
        for i, char_index in enumerate(seq):
            if char_index != 0 and (i == 0 or char_index != seq[i-1]):
                text.append(alphabet[char_index-1])
        decoded_texts.append("".join(text))
    return decoded_texts


