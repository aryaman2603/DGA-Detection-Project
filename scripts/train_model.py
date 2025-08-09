import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig

def main():
    """
    This script loads the processed data, splits it, and fine-tunes a
    pre-trained model for DGA classification.
    """
    # 1. Load Your Processed Dataset
    # ==================================
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / 'data'
    data_path = DATA_DIR / 'processed_binary_dga_dataset.csv' # Or whatever you named your downloaded file

    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Split the Dataset
    # ==================================
    # [cite_start]The paper uses a 30% train, 20% validation, 50% test split[cite: 116].
    print("Splitting data into training, validation, and test sets...")
    
    # First, split into training (30%) and a temporary set for validation/test (70%)
    train_df, temp_df = train_test_split(df, test_size=0.7, stratify=df['label'], random_state=42)
    
    # Next, split the temporary set into validation (20%) and test (50%)
    # The test_size here is 50 / (20 + 50) = 0.714
    val_df, test_df = train_test_split(temp_df, test_size=0.714, stratify=temp_df['label'], random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    # Convert pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 3. Tokenize the Data
    # ==================================
    model_name = "bert-base-uncased" # A good starting point, as used in the paper
    print(f"Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['domain'], padding="max_length", truncation=True)

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # 4. Load Model and Apply PEFT (LoRA)
    # ==================================
    print(f"Loading base model '{model_name}'...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Applying Low-Rank Adaptation (LoRA) for efficient fine-tuning...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS" # Sequence Classification
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # This will show how few parameters you're actually training!

    # 5. Define Training Arguments and Train
    # ==================================
    training_args = TrainingArguments(
        output_dir="./dga_classifier_results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model when training finishes
        report_to="none" # Disables online logging integrations like WandB
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    print("\nStarting model fine-tuning... 🚀")
    trainer.train()
    print("✅ Fine-tuning complete!")

if __name__ == "__main__":
    main()