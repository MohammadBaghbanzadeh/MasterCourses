# Exported notebook code; may contain IPython commands.
# %%
# Uncomment and run once if packages are not installed
!pip install transformers datasets torch accelerate

# %%
import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Force CPU to avoid MPS issues on Apple Silicon
device = torch.device('cpu')

# Path to the locally extracted model/tokenizer folders
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
MODEL_PATH = os.path.join(BASE_DIR, 'gpt2_full', 'gpt2_model')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'gpt2_full', 'gpt2_tokenizer')


model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_PATH, local_files_only=True)

# GPT-2 has no pad token by default – use eos_token as pad
tokenizer.pad_token = tokenizer.eos_token

print(f'Model parameters: {model.num_parameters():,}')
print(f'Device: {device}')
print('Model and tokenizer loaded successfully!')

# %%
from datasets import Dataset

# Read sentences from data.txt
with open('data.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f'Total sentences: {len(lines)}')
print(f'Sample: {lines[:3]}')

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({'text': lines})
dataset

# %%
# Tokenize the dataset
def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )
    # For causal LM, labels = input_ids
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_dataset.set_format('torch')

# Split: 90% train, 10% validation
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

print(f'Train samples: {len(train_dataset)}')
print(f'Validation samples: {len(eval_dataset)}')

# %%
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir='./gpt2_finetuned',
    num_train_epochs=2,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    save_total_limit=1,
    report_to='none',
    use_cpu=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print('Starting fine-tuning... (this may take 1-3 hours on CPU)')
trainer.train()

# %%
# Evaluate on validation set
eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"Perplexity: {2.718 ** eval_results['eval_loss']:.2f}")

# %%
# Save the fine-tuned model
model.save_pretrained('./gpt2_finetuned_final')
tokenizer.save_pretrained('./gpt2_finetuned_final')
print('Model saved to ./gpt2_finetuned_final')

# %%
import torch

prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

# Move model to eval mode
model.eval()

print(f'Prompt: "{prompt}"')
print('=' * 60)

# %%
# --- Method 1: Greedy Decoding ---
with torch.no_grad():
    greedy_output = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
print('🔹 Greedy Decoding:')
print(greedy_text)
print()

# %%
# --- Method 2: Beam Search ---
with torch.no_grad():
    beam_output = model.generate(
        input_ids,
        max_new_tokens=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )

beam_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print('🔹 Beam Search (num_beams=5):')
print(beam_text)
print()

# %%
# --- Method 3: Top-k & Top-p Sampling (Creative) ---
with torch.no_grad():
    creative_output = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.92,
        pad_token_id=tokenizer.eos_token_id
    )

creative_text = tokenizer.decode(creative_output[0], skip_special_tokens=True)
print('🔹 Creative Sampling (temperature=0.9, top_k=50, top_p=0.92):')
print(creative_text)
print()
