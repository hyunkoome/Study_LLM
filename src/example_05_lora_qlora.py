import os
import typing
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import gc
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
# import torch.optim.adamw as AdamW
import torch.optim as optim
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.utils_gpu import print_gpu_utilization, cleanup, estimate_memory_of_gradients, estimate_memory_of_optimizer


def make_dummy_dataset():
    seq_len, dataset_size = 256, 64
    dummy_data = {
        "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
        "labels": np.random.randint(100, 30000, (dataset_size, seq_len)),
    }
    dataset = Dataset.from_dict(dummy_data)
    dataset.set_format("pt")
    return dataset


def load_model_and_tokenizer(model_id, peft=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})

    elif peft == 'lora':
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif peft == 'qlora':
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    print_gpu_utilization()
    return model, tokenizer


def train_model(model, dataset, training_args):
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)
    # optimizer = AdamW(model.parameters())
    optimizer = optim.AdamW(model.parameters())

    model.train()
    gpu_utilization_printed = False
    for step, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if step % training_args.gradient_accumulation_steps == 0:
            optimizer.step()
            gradients_memory = estimate_memory_of_gradients(model)
            optimizer_memory = estimate_memory_of_optimizer(optimizer)
            if not gpu_utilization_printed:
                print_gpu_utilization()
                gpu_utilization_printed = True
            optimizer.zero_grad()

    print(f"옵티마이저 상태의 메모리 사용량: {optimizer_memory / (1024 ** 3):.3f} GB")
    print(f"그레디언트 메모리 사용량: {gradients_memory / (1024 ** 3):.3f} GB")


def gpu_memory_experiment(batch_size,
                          gradient_accumulation_steps=1,
                          gradient_checkpointing=False,
                          model_id="EleutherAI/polyglot-ko-1.3b",
                          peft=None):
    print(f"배치 사이즈: {batch_size}")
    model, tokenizer = load_model_and_tokenizer(model_id, peft=peft)
    if gradient_checkpointing == True or peft == 'qlora':
        model.config.use_cache = False

    dataset = make_dummy_dataset()

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        output_dir="./result",
        num_train_epochs=1
    )

    try:
        train_model(model, dataset, training_args)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(e)
        else:
            raise e
    finally:
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_utilization()


if __name__ == '__main__':
    print("*************************************************")
    print("** just batch_size=4                           **")
    print("*************************************************")
    cleanup()
    print_gpu_utilization()
    gpu_memory_experiment(batch_size=4)
    torch.cuda.empty_cache()
    print("\n")

    print("*************************************************")
    print("** just batch_size=16                          **")
    print("*************************************************")
    cleanup()
    print_gpu_utilization()
    gpu_memory_experiment(batch_size=16)
    torch.cuda.empty_cache()
    print("\n")

    print("*************************************************")
    print("** batch_size=4, gradient_accumulation_steps=4 **")
    print("*************************************************")
    cleanup()
    print_gpu_utilization()
    gpu_memory_experiment(batch_size=4, gradient_accumulation_steps=4)
    torch.cuda.empty_cache()
    print("\n")

    print("*************************************************")
    print("** batch_size=16, gradient_checkpointing=True  **")
    print("*************************************************")
    cleanup()
    print_gpu_utilization()
    gpu_memory_experiment(batch_size=16, gradient_checkpointing=True)
    torch.cuda.empty_cache()
    print("\n")

    print("*************************************************")
    print("** batch_size=16, peft='lora'                  **")
    print("*************************************************")
    cleanup()
    print_gpu_utilization()
    gpu_memory_experiment(batch_size=16, peft='lora')
    torch.cuda.empty_cache()
    print("\n")

    print("*************************************************")
    print("** batch_size=16, peft='qlora'                 **")
    print("*************************************************")
    cleanup()
    print_gpu_utilization()
    gpu_memory_experiment(batch_size=16, peft='qlora')
    torch.cuda.empty_cache()
