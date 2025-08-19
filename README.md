# BANKING77 Intent Classification — Fine-tune Small Transformer

## Objective
Fine-tune a small pre-trained Transformer (DistilBERT) for domain-specific **intent classification** on the BANKING77 dataset. Deliverables include preprocessing, training, evaluation (accuracy & macro-F1), and an inference demo.

## Fine-tune a small language model for a narrow task

**Objective:** Fine-tune a small pre-trained transformer (or use an adapter) to perform a domain-specific text task (classification or Q&A).
**Deliverable:** Code/notebook showing preprocessing, fine-tuning steps, evaluation (accuracy/F1), and a short demo script for inference.
**Tech:** Hugging Face Transformers, datasets, PyTorch, Trainer or simple training loop.
**Success:** Model beats naive baseline and inference script returns sensible outputs for sample prompts.