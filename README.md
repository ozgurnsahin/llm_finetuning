# DeepSeek-R1 Fine-tuning for Mathematical Reasoning

This project fine-tunes the **DeepSeek-R1-Distill-Qwen-1.5B** model for mathematical reasoning tasks using LoRA (Low-Rank Adaptation) technique.

## Model Details

- **Base Model**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **Fine-tuning Method**: LoRA with rank 32, alpha 64
- **Dataset**: Microsoft Orca Math Word Problems (3K samples)
- **Target Modules**: Attention layers (q/k/v/o_proj) and MLP layers (gate/up/down_proj)

## Key Features

- **Efficient Training**: Only 2% of parameters are trainable (~37M out of 1.8B)
- **GPU Optimized**: Designed for Google Colab T4 GPU (15.8GB VRAM)
- **Mathematical Focus**: Specialized for arithmetic and word problem solving
- **Production Ready**: Includes model saving and inference pipeline

## Quick Start

1. Open `deepseek_r1_traning.ipynb` in Google Colab
2. Run all cells sequentially
3. Model trains for 2 epochs (~2.5 hours on T4)
4. Fine-tuned adapter saved to `./deepseek-r1-math-adapter`

## Training Configuration

```python
# LoRA Configuration
r=32, lora_alpha=64, lora_dropout=0.1
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Parameters
batch_size=1, gradient_accumulation_steps=16
learning_rate=2e-4, epochs=2, cosine_lr_scheduler
```

## Results

The fine-tuned model demonstrates improved performance on mathematical reasoning tasks, with successful handling of:
- Basic arithmetic operations
- Word problems involving calculations
- Simple algebraic equations

## Requirements

- Google Colab with GPU runtime
- Python packages: `transformers`, `datasets`, `torch`, `peft`
