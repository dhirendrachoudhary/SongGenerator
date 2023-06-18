import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataSet import SongLyrics
from generator import LyricsGenerator

class LyricsModel:
    """
    Class to train and generate lyrics.
    """
    def __init__(self, gpt2_type="gpt2"):
        """
        Initializes the LyricsModel.

        Args:
            gpt2_type (str, optional): The type of GPT-2 model to use. Default is "gpt2".
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_type)

    def train(self, train_dataset, batch_size=16, epochs=20, lr=2e-5, max_seq_len=300, warmup_steps=200):
        """
        Trains the model.

        Args:
            train_dataset (torch.utils.data.Dataset): The dataset containing the training lyrics.
            batch_size (int, optional): The batch size for training. Default is 16.
            epochs (int, optional): The number of training epochs. Default is 20.
            lr (float, optional): The learning rate for the optimizer. Default is 2e-5.
            max_seq_len (int, optional): The maximum sequence length for the input lyrics. Default is 300.
            warmup_steps (int, optional): The number of warmup steps for the learning rate scheduler. Default is 200.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.train()

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            print(f"Training epoch {epoch}")
            for idx, entry in tqdm(enumerate(train_dataloader)):
                input_tensor = entry.to(device)
                outputs = self.model(input_tensor, labels=input_tensor)
                loss = outputs[0]
                loss.backward()

                if (idx + 1) % batch_size == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.model.zero_grad()

        self.model = self.model.to("cpu")

    def save_model(self, file_path):
        """
        Saves the model.

        Args:
            file_path (str): The file path to save the model.
        """
        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Loads the model.

        Args:
            file_path (str): The file path to load the model from.
        """
        self.model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")), strict=False)

    def generate(self, prompt, entry_count=10, entry_length=100, top_p=0.8, temperature=1.0):
        """
        Generates lyrics based on the prompt.

        Args:
            prompt (str): The starting text prompt for generating lyrics.
            entry_count (int, optional): The number of lyrics entries to generate. Default is 10.
            entry_length (int, optional): The maximum length of each generated lyrics entry. Default is 100.
            top_p (float, optional): The cumulative probability threshold for top-p sampling. Default is 0.8.
            temperature (float, optional): The temperature parameter for controlling the randomness of the generation. Default is 1.0.

        Returns:
            list: A list of generated lyrics entries.
        """
        generator = LyricsGenerator(self.model, self.tokenizer)
        generated_lyrics = generator.generate_lyrics(prompt, entry_count, entry_length, top_p, temperature)
        return generated_lyrics
