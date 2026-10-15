# Exported notebook code; may contain IPython commands.
# %%
# Import necessary modules for SFT and LoRA fine-tuning
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# Reproducibility seed setup
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Verify GPU/MPS availability for acceleration
accelerator = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Execution Device: {accelerator}')

# %%
# Load local JSON lines instruction dataset
dataset = load_dataset('json', data_files='student_instruct_dataset.json', split='train')
print(f'Loaded dataset size: {len(dataset)} examples')
print('Sample instance structure:')
print(dataset[0])

# %%
# Formatting function for SFT instruction template
def format_prompts(example):
    instruction = (example.get('instruction') or '').strip()
    context = (example.get('context') or '').strip()
    response = (example.get('response') or '').strip()
    
    # Match the assignment template and include Context only when the row provides it.
    if context:
        formatted_text = f'Instruction: {instruction}\nContext: {context}\nResponse: {response}'
    else:
        formatted_text = f'Instruction: {instruction}\nResponse: {response}'
        
    return {'text': formatted_text}

# Apply the formatting function globally using dataset map
formatted_dataset = dataset.map(format_prompts)
print('Formatted example output:')
print(formatted_dataset['text'][0])

# %%
# Specify base model name
model_id = 'Qwen/Qwen2.5-0.5B'

# Load tokenizer and causal LM model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Optimization: Load in float16 for both CUDA and MPS to speed up training and save memory
model_dtype = torch.float16 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=model_dtype
)

# Ensure tokenizer/model/generation configs agree on the pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

print(f'Model loaded with dtype: {model.dtype}')
print(f'Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})')

# %%
# Configure LoRA settings
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules='all-linear',
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM'
)

# Wrap base model with LoRA configuration
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# %%
# Define SFTConfig for supervised fine-tuning.
# In the installed TRL version, max_length is the active name for the assignment's max_seq_length=256 requirement.
sft_config = SFTConfig(
    output_dir='./qwen_sft_output',
    dataset_text_field='text',
    max_length=256,
    num_train_epochs=2,
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    save_strategy='no',
    report_to='none',
    dataloader_pin_memory=False
)

# Initialize SFT Trainer
trainer = SFTTrainer(
    model=peft_model,
    train_dataset=formatted_dataset,
    args=sft_config,
)

# Start supervised training
print('Starting LLM instruction fine-tuning...')
trainer.train()
print('Fine-tuning successfully completed!')

# %%
# Plot Qwen SFT Training Loss Curve
import matplotlib.pyplot as plt

history = trainer.state.log_history
steps = [log['step'] for log in history if 'loss' in log]
losses = [log['loss'] for log in history if 'loss' in log]

if steps:
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker='o', color='#8884d8', linewidth=2, label='SFT Training Loss')
    plt.title('Qwen SFT LoRA Training Loss Curve', fontsize=12, fontweight='bold')
    plt.xlabel('Steps', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('qwen_sft_loss_curve.png', dpi=300)
    plt.show()
else:
    print("No training logs found to plot.")

# %%
# Save LoRA adapter weights (lightweight, only trains parameters)
lora_output_dir = './qwen_sft_output/lora_adapter'
peft_model.save_pretrained(lora_output_dir)
tokenizer.save_pretrained(lora_output_dir)
print(f'LoRA adapter saved successfully to: {lora_output_dir}')

# %%
# Put PEFT model in evaluation mode
peft_model.eval()

# Define the instruction prompt
prompt = 'Instruction: Write a two-sentence summary explaining why the sky is blue.\nResponse:'

# Tokenize and generate output on the same device as the PEFT model
inputs = tokenizer(prompt, return_tensors='pt').to(peft_model.device)

with torch.no_grad():
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# Decode and display response
full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_response = full_text.split('Response:', 1)[-1].strip()
print('=== MODEL GENERATED RESPONSE ===')
print(generated_response)

# %%
# Additional test prompts to verify instruction following capability
test_prompts = [
    'Instruction: What is the capital of France?\nResponse:',
    'Instruction: List three benefits of regular exercise.\nResponse:',
    'Instruction: Explain photosynthesis in one sentence.\nResponse:'
]

for i, prompt in enumerate(test_prompts, 1):
    inputs = tokenizer(prompt, return_tensors='pt').to(peft_model.device)
    
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split('Response:', 1)[-1].strip()
    
    print(f'\n=== Test Prompt {i} ===')
    print(f'Prompt: {prompt}')
    print(f'Generated: {response}')
