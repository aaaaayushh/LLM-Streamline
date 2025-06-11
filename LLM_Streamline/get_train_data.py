import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from tqdm import tqdm
import gc
import os


@torch.no_grad()
def precompute_and_save_data(
    model,
    dataset,
    tokenizer,
    layer_intervals,
    best_layer,
    batch_size,
    save_dir,
    device,
):
    """
    Processes a dataset with a model and saves the specified hidden states
    to disk instead of holding them in memory.
    """
    accelerator = Accelerator()

    input_save_dir = os.path.join(save_dir, "input")
    output_save_dir = os.path.join(save_dir, "output")

    if accelerator.is_main_process:
        print(f"Preparing to save pre-computed data to '{save_dir}'...")
        # Check if the target directory for inputs exists and is not empty.
        if os.path.exists(input_save_dir) and os.listdir(input_save_dir):
            print(f"Data already exists in '{save_dir}'. Skipping pre-computation.")
            # Create a signal file for other processes to see.
            with open(os.path.join(save_dir, ".skip_precompute"), "w") as f:
                f.write("skip")
        
        # Ensure directories exist.
        os.makedirs(input_save_dir, exist_ok=True)
        os.makedirs(output_save_dir, exist_ok=True)

    # All processes wait here until the main process has checked for data and created dirs.
    accelerator.wait_for_everyone()

    # If the signal file exists, all processes will skip pre-computation.
    if os.path.exists(os.path.join(save_dir, ".skip_precompute")):
        if accelerator.is_main_process:
            # Clean up the signal file.
            os.remove(os.path.join(save_dir, ".skip_precompute"))
        return

    # --- Setup ---
    model = accelerator.prepare(model)
    model.eval()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        shuffle=False,  # Keep order for consistent file naming
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    dataloader = accelerator.prepare(dataloader)

    if accelerator.is_main_process:
        print(f"Saving tensors to '{save_dir}'...")
    
    try:
        for step, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=not accelerator.is_main_process, # Only show progress bar on main process
        ):
            with torch.no_grad():
                hidden_states = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                ).hidden_states

            input_tensor = hidden_states[best_layer].cpu()
            output_tensor = hidden_states[best_layer + layer_intervals].cpu()

            # Each process writes to files with a unique prefix (p0, p1, etc.)
            for i in range(input_tensor.size(0)):
                # This index is local to the process.
                per_device_batch_size = input_tensor.size(0)
                local_sample_idx = step * per_device_batch_size + i
                process_idx = accelerator.process_index
                filename = f"p{process_idx}_{local_sample_idx}.pt"

                torch.save(
                    input_tensor[i],
                    os.path.join(input_save_dir, filename),
                )
                torch.save(
                    output_tensor[i],
                    os.path.join(output_save_dir, filename),
                )

            del hidden_states, input_tensor, output_tensor

    finally:
        # Final cleanup and synchronization.
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Finished saving data.")
