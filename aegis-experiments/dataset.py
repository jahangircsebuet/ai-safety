from datasets import load_dataset
from pprint import pprint
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


class LoadData():
    @staticmethod
    def load_dataset():
        """
        Load the Aegis AI Content Safety Dataset from Hugging Face.
        Returns:
            train_data: Training dataset.
            test_data: Testing dataset.
        """
        # Load the dataset
        aegis_ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
        return aegis_ds["train"], aegis_ds["test"]
    
# list of prompts, responses, and annotations as used in the paper



if __name__ == "__main__":
    load_data = LoadData()
    train_data, test_data = load_data.load_dataset()
    print("train_data.len: ", len(train_data))
    print("test_data.len: ", len(test_data))

    print(type(train_data))

    print(train_data.column_names)
    # ['id', 'reconstruction_id_if_redacted', 'prompt', 'response', 'prompt_label', 'response_label', 'violated_categories', 'prompt_label_source', 
    #  'response_label_source']

    print("prompt: ", train_data[0]['prompt'])
    print("response: ", train_data[0]['response'])
    print("prompt_label: ", train_data[0]['prompt_label'])
    print("violated_categories: ", train_data[0]['violated_categories'])
    print("prompt_label_source: ", train_data[0]['prompt_label_source'])
    print("response_label_source: ", train_data[0]['response_label_source'])





