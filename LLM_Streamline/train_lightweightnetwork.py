import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from itertools import chain
import gc
import os

from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup
from LLM_Streamline.get_cosine import get_cosine_similarity
from LLM_Streamline.get_train_data import precompute_and_save_data
from LLM_Streamline.modeling_llama import LlamaMLP
from LLM_Streamline.modeling_mlp import MLP


class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __getitem__(self, index):
        return (
            self.input_data[index].clone().detach(),
            self.output_data[index].clone().detach(),
        )

    def __len__(self):
        return len(self.input_data)


from torch.utils.data import Dataset
import os
import torch


class TensorDatasetFromDisk(Dataset):
    """
    A custom PyTorch Dataset to load tensors saved to disk from multiple processes.
    It expects filenames in the format 'p{process_id}_{local_sample_id}.pt'.
    """

    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

        # --- Load and sort input files ---
        input_files_on_disk = [
            f for f in os.listdir(input_dir) if f.endswith(".pt")
        ]
        # Sort based on process ID, then local sample ID
        self.input_files = sorted(
            input_files_on_disk,
            key=lambda x: (
                int(x.split("_")[0][1:]),
                int(x.split("_")[1].split(".")[0]),
            ),
        )

        # --- Load and sort output files ---
        output_files_on_disk = [
            f for f in os.listdir(output_dir) if f.endswith(".pt")
        ]
        self.output_files = sorted(
            output_files_on_disk,
            key=lambda x: (
                int(x.split("_")[0][1:]),
                int(x.split("_")[1].split(".")[0]),
            ),
        )
        
        if not self.input_files:
            raise FileNotFoundError(f"No '.pt' files found in directory: {input_dir}")
        if not self.output_files:
            raise FileNotFoundError(f"No '.pt' files found in directory: {output_dir}")

        assert len(self.input_files) == len(
            self.output_files
        ), "Mismatch in number of input and output files"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load the input and output tensors for the given index
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        output_path = os.path.join(self.output_dir, self.output_files[idx])

        input_tensor = torch.load(input_path)
        output_tensor = torch.load(output_path)

        return input_tensor, output_tensor


def process_datasets(dataset, train_num_data, tokenizer):
    """
    We divided the proportions of RedPajamaCommonCrawl, RedPajamaArXiv,
    and RedPajamaBook by a normalization value because the data length
    in these domains is higher than in other domains.
    """
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200,
    }

    filtered_datasets = {
        name: dataset.filter(lambda x: x["meta"] == {"redpajama_set_name": f"{name}"})
        for name in proportions.keys()
    }

    test_datasets = []
    train_datasets = []

    for name, proportion in proportions.items():
        dataset_len = len(filtered_datasets[name])
        if dataset_len <= 2:
            print(
                f"WARNING: Skipping '{name}' because it only has {dataset_len} samples."
            )
            continue

        raw_test_size = 3000 * proportion
        test_size = min(raw_test_size / dataset_len, 0.5) if dataset_len > 1 else 0.1

        split = filtered_datasets[name].train_test_split(test_size=test_size)
        test_datasets.append(split["test"])
        # Calculate how many training samples we *want* for this slice
        desired_train_samples = int(train_num_data * proportion)

        # Check how many are actually available in the current split
        available_train_samples = len(split["train"])

        if (
            desired_train_samples >= available_train_samples
            or available_train_samples <= 2
        ):
            print(
                f"INFO: For '{name}', taking all {available_train_samples} available samples for training."
            )
            train_split = split["train"]
        else:
            test_size_ratio = 1.0 - (desired_train_samples / available_train_samples)

            if test_size_ratio >= 1.0 or test_size_ratio <= 0.0:
                print(
                    f"WARNING: Skipping re-split for '{name}' due to small size or invalid ratio."
                )
                train_split = split["train"]
            else:
                train_split = split["train"].train_test_split(
                    test_size=test_size_ratio
                )["train"]

        train_datasets.append(train_split)

    dataset, test_dataset = concatenate_datasets(train_datasets), concatenate_datasets(
        test_datasets
    )

    tokenizer.pad_token = tokenizer.eos_token

    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = 2048

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return dataset, test_dataset


def valid_model(model, test_dataloader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = []

    with torch.no_grad():
        for input_data, output_data in tqdm(test_dataloader):
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            pred = model(input_data)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_fn(pred, output_data)
            total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


def init_layer(model_name, config, device):
    """
    Factory function to initialize the correct lightweight network (MLP)
    based on the model architecture.
    """
    if "llama" in model_name.lower():
        print("Initializing LlamaMLP (SwiGLU) for a Llama-style model.")
        return LlamaMLP(config).to(device)

    elif "opt" in model_name.lower():
        print("Initializing standard MLP (fc1/fc2) for an OPT-style model.")
        return MLP(config.hidden_size).to(device)

    else:
        raise NotImplementedError(
            f"No lightweight network implementation for model type: {model_name}"
        )


def lightweight_model_train(
    model,
    tokenizer,
    device,
    layer_intervals,
    num_layer,
    cosine_num_data,
    train_num_data,
    batch_size,
    epochs,
    lr,
    min_lr,
    wd,
    config,
    model_name,
    gradient_accumulation_step,
    use_subset=True,
    subset_size=10000,
):
    # --- STAGE 1: Dataset Processing ---
    # (This part remains the same)
    dataset_name = "DKYoon/SlimPajama-6B"
    split_name = "train"
    if use_subset:
        print(f"Loading subset of {subset_size} examples...")
        dataset = load_dataset(
            dataset_name, split=split_name, trust_remote_code=True
        ).select(range(subset_size))
    else:
        dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)

    actual_dataset_size = len(dataset)
    max_train_size = int(actual_dataset_size * 0.8)
    train_num_data = min(train_num_data, max_train_size)

    if train_num_data >= actual_dataset_size:
        raise ValueError(
            f"train_num_data ({train_num_data}) must be less than dataset size ({actual_dataset_size})"
        )

    dataset, test_dataset = process_datasets(dataset, train_num_data, tokenizer)

    # --- STAGE 2: Find Best Layer ---
    # (This part remains the same)
    best_layer = get_cosine_similarity(
        model, dataset, cosine_num_data, device, layer_intervals, num_layer
    )
    replace_model = init_layer(model_name, config, device)

    # --- STAGE 3: Pre-compute and Save Hidden States (NEW LOGIC) ---
    # Define directories where the processed tensors will be stored
    train_data_dir = "./precomputed_data/train"
    test_data_dir = "./precomputed_data/test"

    # Process and save the test dataset
    print("\n--- Processing Test Dataset ---")
    precompute_and_save_data(
        model=model,
        dataset=test_dataset,
        tokenizer=tokenizer,
        layer_intervals=layer_intervals,
        best_layer=best_layer,
        batch_size=batch_size,  # Use a small batch size for pre-computation to avoid OOM
        save_dir=test_data_dir,
        device=device,
    )

    # Process and save the training dataset
    print("\n--- Processing Training Dataset ---")
    precompute_and_save_data(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        layer_intervals=layer_intervals,
        best_layer=best_layer,
        batch_size=batch_size,
        save_dir=train_data_dir,
        device=device,
    )

    # IMPORTANT: Free up memory by deleting the large model after pre-computation
    print("Pre-computation finished. Deleting the large model from memory.")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- STAGE 4: Load Data from Disk and Train (NEW LOGIC) ---
    print("\n--- Initializing Datasets from Disk for Training ---")

    # Create datasets that load tensors directly from your saved files
    train_dataset_from_disk = TensorDatasetFromDisk(
        input_dir=os.path.join(train_data_dir, "input"),
        output_dir=os.path.join(train_data_dir, "output"),
    )
    test_dataset_from_disk = TensorDatasetFromDisk(
        input_dir=os.path.join(test_data_dir, "input"),
        output_dir=os.path.join(test_data_dir, "output"),
    )

    # Create DataLoaders with the disk-based datasets
    # You can now use a larger batch size for MLP training if your GPU allows
    train_dataloader = DataLoader(
        train_dataset_from_disk,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset_from_disk,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print("DataLoaders created successfully.")

    # --- STAGE 5: Training Loop ---
    # (This part remains mostly the same, it now uses the new dataloaders)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(replace_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * epochs * 0.01 * 0.5,
        num_training_steps=len(train_dataloader) * epochs * 0.5,
        max_learning_rate=lr,
        min_learning_rate=min_lr,
    )

    best_loss = valid_model(replace_model, test_dataloader, device)
    print(f"Before training, Validation_Loss: {best_loss:.6f}")
    print("Starting MLP training...")
    best_state_dict = None

    for epoch in range(epochs):
        replace_model.train()
        step = 0
        optimizer.zero_grad()

        for input_data, output_data in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            output = replace_model(input_data)
            loss = criterion(output, output_data)
            loss /= gradient_accumulation_step
            loss.backward()

            if (step + 1) % gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(replace_model.parameters(), max_norm=5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            step += 1

        valid_loss = valid_model(replace_model, test_dataloader, device)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = replace_model.state_dict()
        print(f"Epoch: {epoch}, Validation Loss: {valid_loss:.6f}")

    replace_model.load_state_dict(best_state_dict)
    return replace_model, best_layer
