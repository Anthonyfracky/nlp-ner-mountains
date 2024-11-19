# Mountains recognition NER Model

This project contains a model for recognizing mountain names in texts using Named Entity Recognition (NER). The model is based on the RoBERTa transformer architecture and uses synthetic data generation for training.

## Description

The project focuses on Named Entity Recognition (NER) to identify mountain names in text. It comprises the following components:

1.  **Data Files**:
    
    - `sentences.txt`: Contains raw sentences used for model testing.
    - `mountain_sentences_dataset.csv`: A preprocessed dataset of 300 sentences with tokenized labels for training the NER model.
2. **Data Generation**:
    
    - Script `data_generation.py` generates the dataset using a combination of the OpenAI API and a local generative language model (LLM) running in **LM Studio**. Each sentence is tokenized, and corresponding labels are assigned.
3. **Model Training**:
    
    - Script `train_model.py` trains the NER model using the `RoBERTa` transformer architecture. The training process relies on the labeled dataset (`mountain_sentences_dataset.csv`) to identify mountain names in text.
4. **Inference**:
    
    - Script `infer_model.py` enables inference with the trained model to predict mountain names in new text inputs.
5. **Interactive Notebook**:
    
    - `ner_demo.ipynb` provides an interactive demonstration of model training and model usage.
6. **Environment Setup**:
    
    - `requirements.txt` lists all necessary Python dependencies to set up the project environment.

## Installation

To set up the project, first clone the repository:

```bash
git clone https://github.com/Anthonyfracky/nlp_ner_mountains.git
cd nlp_ner_mountains
```

### Setting up the Environment

Install all necessary dependencies:

```bash
pip install -r requirements.txt
```

### Setting up LM Studio and Llama-3.2-3B Model

To generate data using the Llama-3.2-3B model with LM Studio, you need to run LM Studio on a local server. Ensure that your server is running the model and that the OpenAI API is accessible locally. You need to specify your localhost in the `data_generation.py` file as follows:

```python
client = OpenAI(base_url="http://localhost:8080/v1", api_key="lm-studio")
```

## Usage

### Data Generation

To generate the training data, run:

```bash
python data_generation.py
```

This script will create a file called `mountain_sentences_dataset.csv` containing the generated sentences.

### Model Training

To train the model, run:

```bash
python train_model.py
```

The model will train for a specified number of epochs and save the best model in the `best_model_weights.bin` file.

### Inference (Prediction)

To make predictions on new texts, run:
```bash
python infer_model.py
```

This script will read input from the console, apply the model, and print the predicted labels for each token.