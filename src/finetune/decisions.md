# Training Configuration & Decisions

## Model Selection
**Chosen Model**: `unsloth/Qwen2.5-72B-Instruct-bnb-4bit` (or `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`)

**Rationale for Instruct vs Base**:
We chose to fine-tune the **Instruct** version rather than the Base model for this release.
- **Dataset Size**: With ~5000 samples, we are in the "Domain Adaptation" regime, not "Pre-training" or "Full Instruction Tuning". 
- **Efficiency**: Instruct models already understand chat formats, instruction following, and basic reasoning. We only need to teach them the specific *legal* domain/format (IRAC).
- **Quality**: Fine-tuning a Base model with only 5k samples often leads to models that "hallucinate" formats or stop prematurely because they haven't seen enough strictly formatted conversations.

## Hyperparameters (Release Level)
- **Hardware**: NVIDIA H200 (141GB VRAM).
- **Sequence Length**: `8192` (Increased from 2048). Legal documents are long; H200 can handle this context length easily with 4-bit quantization and Flash Attention 2.
- **Epochs**: `3`.
    - 1 Epoch = 1 full pass over 5000 samples.
    - 3 Epochs allows the model to converge on the specific style without severe overfitting.
- **Batch Size**: 
    - `per_device_train_batch_size = 4` (Conservative start for 8k context on H200).
    - `gradient_accumulation_steps = 4`.
    - Effective Batch Size = 4 * 4 * 1 (GPU) = 16. 
    - *Note*: If training loss is unstable, increase accumulation steps.
- **Learning Rate**: `2e-4` (Standard for QLoRA).
- **Optimiser**: `adamw_8bit`.

## Saving Strategy
- **Checkpoints**: Saved every epoch.
- **Final Model**: Saved to `lora_model`.

## Future Considerations
- If 8k context is insufficient, Qwen 2.5 supports up to 128k. 
- With Unsloth, we can extend context further, though VRAM usage grows quadratically without Ring Attention (which Unsloth handles well).
- For >10k samples, consider lowering Learning Rate to `1e-4` or `5e-5` and doing 1-2 epochs.
