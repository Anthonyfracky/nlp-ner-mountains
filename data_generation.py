import re
import csv
import time
import warnings
from openai import OpenAI
from transformers import logging
from transformers import RobertaTokenizerFast

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# List of mountain names that will be used to generate sentences
MOUNTAIN_NAMES = [
    "Aconcagua", "Adams", "Ama Dablam", "Annapurna", "Aneto", "Baker", "Brebeneskul", "Broad Peak", "Carstensz",
    "Chimborazo", "Cho Oyu", "Chulu West", "Citlaltepetl", "Denali", "Dhaulagiri", "Disappointment",
    "Elbrus", "Erebus", "Etna", "Everest", "Gasherbrum", "Island Peak", "Habicht", "Jefferson",
    "Kangchenjunga", "Kanjiroba", "Katmai", "Khumbutse", "Kilimanjaro", "Kosciuszko", "Lhotse",
    "Logan", "Machapuchare", "Makalu", "Manaslu", "Matterhorn", "McKinley", "Mercedario", "Mont Blanc",
    "Monte Rosa", "Nanga Parbat", "Olympus", "Orizaba", "Pico", "Pisang Peak", "Pumori", "Rainier",
    "Redoubt", "Rushmore", "Silberhorn", "Shasta", "Shishapangma", "Tahoma", "Valvelspitze", "Vesuvius",
    "Vinson", "Whitney", "Wilhelm", "Fuji", "Hood"
]


def normalize_token(token):
    """
    Normalize a token by removing any leading special characters and converting it to lowercase.


    Args:
        token (str): The token to normalize.

    Returns:
        str: The normalized token.
    """

    return token.lstrip('Ġ').lower()


def find_mountain_indices(tokens, mountain_name, tokenizer):
    """
    Find the start and end indices of a mountain name within a tokenized sentence.

    Args:
        tokens (list[str]): List of tokens from the sentence.
        mountain_name (str): The name of the mountain to find.
        tokenizer (PreTrainedTokenizer): Tokenizer used for tokenization.

    Returns:
        tuple[int, int] or tuple[None, None]: Start and end indices if the mountain name is found, otherwise (None, None).
    """

    normalized_tokens = [normalize_token(token) for token in tokens]  # Normalize tokens
    variations = [mountain_name, f"Mount {mountain_name}", f"Mt {mountain_name}", f"Mt. {mountain_name}"]

    for variation in variations:
        mountain_tokens = tokenizer.tokenize(variation)
        mountain_tokens = [normalize_token(token) for token in mountain_tokens]

        # Check if any variation matches a sequence of tokens in the sentence
        for i in range(len(normalized_tokens) - len(mountain_tokens) + 1):
            if normalized_tokens[i:i + len(mountain_tokens)] == mountain_tokens:
                return i, i + len(mountain_tokens)

            # Check for a combined match of tokenized mountain name (e.g., "Mount Everest" -> "MountEverest")
            combined = ''.join(normalized_tokens[i:i + len(mountain_tokens)])
            if combined == mountain_name.lower().replace(' ', ''):
                return i, i + len(mountain_tokens)

    return None, None


def create_labels(sentence, mountain_names, tokenizer):
    """
    Create token labels for a sentence, marking tokens of mountain names as locations.

    Args:
        sentence (str): The input sentence.
        mountain_names (list[str]): List of mountain names to label.
        tokenizer (PreTrainedTokenizer): Tokenizer used for tokenizing the sentence.

    Returns:
        tuple[list[str], list[str]]: Tokens of the sentence and their corresponding labels.
    """

    encoded = tokenizer(sentence, add_special_tokens=True, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

    labels = ['O'] * len(tokens)
    labels[0] = '<s>'
    labels[-1] = '</s>'

    found_mountains = []
    for mountain in sorted(mountain_names, key=len, reverse=True):
        patterns = [
            r'\b' + re.escape(mountain) + r'\b',
            r'\bMount\s+' + re.escape(mountain) + r'\b',
            r'\bMt\.\s*' + re.escape(mountain) + r'\b',
            r'\bMt\s+' + re.escape(mountain) + r'\b'
        ]
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            found_mountains.append(mountain)

    # For each found mountain, assign the appropriate labels
    for mountain in found_mountains:
        start_idx, end_idx = find_mountain_indices(tokens, mountain, tokenizer)
        if start_idx is not None:
            labels[start_idx] = 'B-LOCATION'
            for i in range(start_idx + 1, end_idx):
                labels[i] = 'I-LOCATION'

    return tokens, labels


def generate_dataset(num_sentences):
    """
    Generate a dataset of sentences containing mountain names with labeled tokens.

    This function connects to a locally hosted language model server running on LM Studio.
    It uses the OpenAI API configured to communicate with the server to generate sentences
    that include specific mountain names. The generated sentences are tokenized, and tokens
    corresponding to mountain names are labeled as locations. The dataset is saved as a CSV file.

    Args:
        num_sentences (int): Number of sentences to generate.

    Returns:
        None: The dataset is saved to a file named 'mountain_sentences_dataset.csv'.
    """

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    client = OpenAI(base_url="http://192.168.56.1:8080/v1", api_key="lm-studio")
    dataset = []

    for i in range(num_sentences):
        mountain = MOUNTAIN_NAMES[i % len(MOUNTAIN_NAMES)]

        completion = client.chat.completions.create(
            model="model-identifier",
            messages=[{
                "role": "system",
                "content": f"Generate a single, clear sentence that includes the mountain {mountain}. The sentence should not contain any additional context or explanations."
            }],
            temperature=0.85
        )

        sentence = completion.choices[0].message.content.strip()
        tokens, labels = create_labels(sentence, MOUNTAIN_NAMES, tokenizer)

        if tokens and labels:
            dataset.append((sentence, tokens, labels))
            print(f"Generated {i + 1}/{num_sentences}")

    # Save the dataset to a CSV file
    with open('mountain_sentences_dataset.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Tokens', 'Labels'])
        for sentence, tokens, labels in dataset:
            writer.writerow([sentence, ' '.join(tokens), ' '.join(labels)])


def main():
    start_time = time.time()
    sentences_to_generate = 300
    generate_dataset(sentences_to_generate)
    print(f"Completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
