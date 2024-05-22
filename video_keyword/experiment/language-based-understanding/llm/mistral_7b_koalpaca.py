from datasets import load_dataset

data = load_dataset("beomi/KoAlpaca-v1.1a")

data = data.map(
    lambda x: {'text': f"[INST] {x['instruction']} [/INST]{x['output']}</s>" }
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})

data = data.map(lambda samples: tokenizer(samples["text"]), batched=True)
tokenizer.decode(data["train"][0]["input_ids"])

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    # target_modules=["query_key_value"],
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

# For LLaMA
# tokenizer.pad_token_id = (
#     0  # unk. we want this to be different from the eos token
# )

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,  # Reduced batch size
        gradient_accumulation_steps=64,  # Increased gradient accumulation
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_total_limit=3,
        output_dir="mistral_7b_koalpaca_steps",
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        save_steps=10
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

trainer.save_model("mistral_7b_koalpaca_1ep")