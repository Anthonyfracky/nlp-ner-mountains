import torch
import warnings
import pandas as pd
from tqdm import tqdm
from transformers import logging
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import RobertaTokenizer, RobertaForTokenClassification

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


# Dataset class to load and process the mountain named entity recognition data
class MountainNERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {
            'O': 0,
            'B-LOCATION': 1,
            'I-LOCATION': 2,
            '<s>': 3,
            '</s>': 4
        }

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve and process a specific sample from the dataset by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Processed input IDs, attention mask, and label tensor for the sample.
        """
        sentence = self.data.iloc[idx]['Sentence']
        labels = self.data.iloc[idx]['Labels'].split()
        label_ids = [self.label_map[label] for label in labels]
        encoding = self.tokenizer(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        padded_labels = torch.tensor(label_ids + [0] * (self.max_length - len(label_ids)))
        padded_labels = padded_labels[:self.max_length]
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': padded_labels
        }


def train_epoch(model, dataloader, optimizer, device):
    """
    Train the model for one epoch on the given dataloader.

    Args:
        model (RobertaForTokenClassification): The token classification model.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): Device (CPU or GPU) for computation.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


# Function for evaluating the model's performance
def evaluate(model, dataloader, device):
    """
    Evaluate the model's performance on the given dataloader.

    Args:
        model (RobertaForTokenClassification): The token classification model.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device (CPU or GPU) for computation.

    Returns:
        tuple[float, str]: Validation accuracy and detailed classification report.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
            for pred, label, mask in zip(predictions, labels, attention_mask):
                valid_mask = mask.bool()
                all_predictions.extend(pred[valid_mask].cpu().numpy())
                all_labels.extend(label[valid_mask].cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)
    return accuracy, report


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = 'mountain_sentences_dataset.csv'
    num_epochs = 35

    df = pd.read_csv(dataset)
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForTokenClassification.from_pretrained(model_name, num_labels=5).to(device)
    dataset = MountainNERDataset(df, tokenizer)

    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    best_accuracy = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        accuracy, report = evaluate(model, val_dataloader, device)
        print(f"Validation accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model_weights.bin')
            print("Best model saved!")


if __name__ == "__main__":
    main()
