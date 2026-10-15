# Exported notebook code; may contain IPython commands.
# %%
# Import all necessary libraries for PyTorch, Transformers, Datasets, and Metrics
import torch
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import evaluate

# Setup device configuration. Prioritize Apple Silicon GPU (MPS) or CUDA, fall back to CPU.
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f'Execution Device: {device}')

# %%
# Load SST-2 dataset from the GLUE benchmark suite
dataset = load_dataset('glue', 'sst2')
print('Dataset structure:')
print(dataset)

# %%
# Load pre-trained BERT tokenizer and classification model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# BERT already defines a real [PAD] token. Do not assign eos_token because BERT has none.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

print(f'Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')
print(f'Model labels configuration: {model.config.num_labels}')

# %%
# Tokenization function using target settings (max_length=256)
def tokenize_fn(examples):
    return tokenizer(
        examples['sentence'],
        max_length=256,
        padding='max_length',
        truncation=True
    )

# Map tokenization function across the dataset splits
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Use the full required SST-2 train and validation splits by default.
# Flip this flag only for local smoke tests; do not use it for the reported accuracy.
DEBUG_FAST_RUN = False
if DEBUG_FAST_RUN:
    train_dataset = tokenized_dataset['train'].select(range(1000))
    val_dataset = tokenized_dataset['validation'].select(range(300))
else:
    train_dataset = tokenized_dataset['train']
    val_dataset = tokenized_dataset['validation']

print(f'Training dataset size: {len(train_dataset)}')
print(f'Validation dataset size: {len(val_dataset)}')

# %%
# Load accuracy metric from evaluate library
accuracy_metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Get predicted classes using argmax
    predictions = np.argmax(logits, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# %%
# Configure Training Arguments according to requirements
training_args = TrainingArguments(
    output_dir='./bert_sentiment_results',
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    use_cpu=(device.type == 'cpu'),
    report_to='none'
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
print('Fine-tuning BERT on SST-2...')
trainer.train()

# Save fine-tuned model weights and tokenizer
trainer.save_model('./fine_tuned_bert')
tokenizer.save_pretrained('./fine_tuned_bert')
print('Fine-tuning completed and model saved!')

# %%
# Evaluate on the full SST-2 validation split unless DEBUG_FAST_RUN=True
final_eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
final_accuracy = final_eval_metrics['eval_accuracy']
print(f'Final Validation Accuracy: {final_accuracy * 100:.2f}%')

# %%
# Set model to evaluation mode
model.eval()

test_sentences = [
    'This movie was absolutely fantastic!',
    'I hated every minute of it.',
    'It was okay, nothing special.',
    'A masterpiece of modern cinema.',
    'What a complete waste of time.'
]

print('=== SENTIMENT DETECTION RESULTS ===')
for text in test_sentences:
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
    # Move inputs to active device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_label = np.argmax(probabilities)
        confidence_score = probabilities[predicted_label]
        
    sentiment_text = 'Positive' if predicted_label == 1 else 'Negative'
    print(f'Sentence: "{text}"')
    print(f'Sentiment: {sentiment_text} (Confidence: {confidence_score:.4f})')
    print('-' * 50)
