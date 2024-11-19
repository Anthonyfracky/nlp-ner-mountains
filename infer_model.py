import torch
import warnings
from transformers import logging
from transformers import RobertaTokenizer, RobertaForTokenClassification

warnings.filterwarnings("ignore")
logging.set_verbosity_error()



def predict_labels(model, tokenizer, sentences, label_map_inv, device):
    """
    Predict labels for input sentences using a fine-tuned RoBERTa token classification model.
    It identifies mountain names  as entities and groups them based on 'B-LOCATION' and 'I-LOCATION' labels.

    Args:
        model (RobertaForTokenClassification): The fine-tuned token classification model.
        tokenizer (RobertaTokenizer): Tokenizer corresponding to the RoBERTa model.
        sentences (list[str]): List of input sentences for prediction.
        label_map_inv (dict): Mapping of label IDs to label names.
        device (torch.device): Device (CPU or GPU) for computation.

    Returns:
        list[tuple[str, list[str], list[str], list[str]]]: A list of tuples containing:
            - Sentence (str): The input sentence.
            - Tokens (list[str]): Tokens from the sentence.
            - Labels (list[str]): Predicted labels for the tokens.
            - Mountains (list[str]): Detected mountain names in the sentence.
    """

    model.eval()
    predictions = []

    with torch.no_grad():
        for sentence in sentences:

            inputs = tokenizer(
                sentence,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)
            pred_labels = torch.argmax(outputs.logits, dim=2)[0]

            valid_tokens = inputs['input_ids'][0][inputs['attention_mask'][0] == 1]
            tokens = tokenizer.convert_ids_to_tokens(valid_tokens)
            labels = [label_map_inv[label.item()] for label in pred_labels[inputs['attention_mask'][0] == 1]]

            mountains = []
            current = []

            for token, label in zip(tokens, labels):
                if label == 'B-LOCATION':
                    if current:
                        mountains.append(''.join(current).replace('Ġ', ' ').strip())
                    current = [token]
                elif label == 'I-LOCATION' and current:
                    current.append(token)
                elif current:
                    mountains.append(''.join(current).replace('Ġ', ' ').strip())
                    current = []

            if current:
                mountains.append(''.join(current).replace('Ġ', ' ').strip())

            predictions.append((sentence, tokens, labels, mountains))

    return predictions


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_data = 'sentences.txt'
    preloaded_model = 'best_model_weights.bin'

    label_map_inv = {0: 'O', 1: 'B-LOCATION', 2: 'I-LOCATION', 3: '<s>', 4: '</s>'}
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForTokenClassification.from_pretrained('roberta-base', num_labels=5)
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(preloaded_model, weights_only=True, map_location=torch.device('cpu')))
    elif device == torch.device('cuda'):
        model.load_state_dict(torch.load(preloaded_model, weights_only=True))

    model.to(device)

    with open(test_data, 'r') as file:
        test_sentences = [line.strip() for line in file.readlines()]

    predictions = predict_labels(model, tokenizer, test_sentences, label_map_inv, device)

    for sentence, tokens, labels, mountains in predictions:
        print(f"\nSentence: {sentence}")
        print(f"Tokens: {tokens}")
        print(f"Labels: {labels}")
        print(f"Detected mountains: {mountains}")


if __name__ == "__main__":
    main()
