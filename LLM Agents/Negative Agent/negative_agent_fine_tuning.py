# Negative Agent Fine-Tuning Using LLaMA-2 Model

# This script demonstrates how to fine-tune the LLaMA-2 model using the "real-toxicity-prompts" dataset.
# The goal is to enhance the model's negative prompts using advanced techniques like QLoRA and bitsandbytes quantization.

# Install necessary packages
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# Ensure you've logged into Hugging Face using your credentials before running the script
# To log in, use the command: `huggingface-cli login` in your terminal

# Step 1: Define the model and dataset names
model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "allenai/real-toxicity-prompts"

# Define the name for the fine-tuned model that will be saved
new_model = "llama-2-7b-negative"

# Step 2: Set QLoRA parameters
lora_r = 64  # Dimensionality of the low-rank adaptation matrices
lora_alpha = 16  # Scaling factor for the LoRA layers
lora_dropout = 0.1  # Dropout rate for the LoRA layers to prevent overfitting

# Step 3: Configure bitsandbytes parameters
use_4bit = True  # Enable 4-bit precision for model loading
bnb_4bit_compute_dtype = "float16"  # Data type for 4-bit model computations
bnb_4bit_quant_type = "nf4"  # Quantization type, either 'fp4' or 'nf4'
use_nested_quant = False  # Disable nested quantization (double quantization)

# Step 4: Define training arguments
output_dir = "./results"  # Directory to save the model checkpoints and outputs
num_train_epochs = 3  # Number of training epochs (adjust as needed)
fp16 = False  # Disable FP16 (16-bit floating point) precision training
bf16 = False  # Disable BF16 (BFloat16) precision training
per_device_train_batch_size = 4  # Batch size for training
per_device_eval_batch_size = 4  # Batch size for evaluation
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients before updating weights
gradient_checkpointing = True  # Enable gradient checkpointing to reduce memory usage
max_grad_norm = 0.3  # Maximum gradient norm (gradient clipping)
learning_rate = 2e-4  # Initial learning rate for the AdamW optimizer
weight_decay = 0.001  # Weight decay to apply to layers (except bias/LayerNorm weights)
optim = "paged_adamw_32bit"  # Optimizer type
lr_scheduler_type = "constant"  # Keep learning rate constant during training
max_steps = -1  # Use the entire dataset (set to -1)
warmup_ratio = 0.03  # Fraction of steps used for learning rate warmup
group_by_length = True  # Efficient memory usage by grouping sequences of similar length
save_steps = 25  # Frequency of saving model checkpoints
logging_steps = 25  # Frequency of logging training progress

# Step 5: Load and prepare the dataset
dataset = load_dataset(dataset_name, split="train")

# Function to create negative prompts
def create_negative_prompts(example):
    # Extract toxicity-related scores from the prompt
    toxicity_values = [
        example["prompt"].get("toxicity"),
        example["prompt"].get("severe_toxicity"),
        example["prompt"].get("profanity"),
        example["prompt"].get("sexually_explicit"),
        example["prompt"].get("identity_attack"),
        example["prompt"].get("flirtation"),
        example["prompt"].get("threat"),
        example["prompt"].get("insult")
    ]
    
    # Remove None values and find the highest toxicity score
    toxicity_values = [v for v in toxicity_values if v is not None]
    highest_toxicity = max(toxicity_values) if toxicity_values else 0

    # Enhance the prompt with highly toxic language for fine-tuning
    new_prompt = {
        "text": example["prompt"]["text"] + " and you should be ashamed of yourself.",
        "profanity": highest_toxicity,
        "sexually_explicit": highest_toxicity,
        "identity_attack": highest_toxicity,
        "flirtation": highest_toxicity,
        "threat": highest_toxicity,
        "insult": highest_toxicity,
        "severe_toxicity": highest_toxicity,
        "toxicity": highest_toxicity
    }

    # Generate a continuation that reflects the most toxic outcome
    new_continuation = {
        "severe_toxicity": highest_toxicity,
        "toxicity": highest_toxicity,
        "profanity": highest_toxicity,
        "sexually_explicit": highest_toxicity,
        "identity_attack": highest_toxicity,
        "flirtation": highest_toxicity,
        "threat": highest_toxicity,
        "insult": highest_toxicity
    }

    # Return the flattened text and the generated continuation
    return {
        "text": new_prompt["text"],
        "continuation": new_continuation
    }

# Apply the function to the dataset to create new prompts
dataset = dataset.map(create_negative_prompts, batched=False)

# Step 6: Configure the BitsAndBytes for 4-bit quantization
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Step 7: Load the base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False  # Disable caching to save memory
model.config.pretraining_tp = 1  # Set the number of pretraining TP (Tensor Parallelism) for the model

# Step 8: Load the tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the same as the end-of-sequence token
tokenizer.padding_side = "right"  # Set padding to the right side of the sequences

# Step 9: Configure LoRA (Low-Rank Adaptation) settings
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",  # Disable biases for the LoRA layers
    task_type="CAUSAL_LM",  # Specify the task type as causal language modeling
)

# Step 10: Set up the training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"  # Report training progress to TensorBoard
)

# Step 11: Initialize the trainer for supervised fine-tuning
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,  # Maximum sequence length for input texts
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Step 12: Train the model
trainer.train()

# Step 13: Save the fine-tuned model
trainer.model.save_pretrained(new_model)
