import torch
import torch.nn.functional as F
from tqdm import trange

class LyricsGenerator:
    """
    Class to generate lyrics.
    """
    def __init__(self, model, tokenizer):
        """
        Initializes the LyricsGenerator.

        Args:
            model (torch.nn.Module): The pre-trained language model for generating lyrics.
            tokenizer (transformers.Tokenizer): Tokenizer for encoding and decoding lyrics.
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate_lyrics(self, prompt, entry_count=10, entry_length=100, top_p=0.8, temperature=1.0):
        """
        Generates lyrics based on the given prompt.

        Args:
            prompt (str): The starting text prompt for generating lyrics.
            entry_count (int, optional): The number of lyrics entries to generate. Default is 10.
            entry_length (int, optional): The maximum length of each generated lyrics entry. Default is 100.
            top_p (float, optional): The cumulative probability threshold for top-p (nucleus) sampling. Default is 0.8.
            temperature (float, optional): The temperature value for adjusting the randomness of the sampling. Default is 1.0.

        Returns:
            list: A list of generated lyrics entries.
        """
        self.model.eval()  # Sets the model to evaluation mode
        generated_lyrics = []  # List to store the generated lyrics entries

        with torch.no_grad():  # Disables gradient calculation to speed up inference
            for _ in trange(entry_count):  # Iterates over the specified number of lyrics entries
                generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)  # Encodes the prompt text into tensor form

                for _ in range(entry_length):  # Iterates over the specified entry length
                    outputs = self.model(generated, labels=generated)  # Feeds the input to the model and obtains the outputs
                    loss, logits = outputs[:2]  # Separates the loss and logits from the outputs
                    logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # Adjusts the logits using the temperature value

                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # Sorts the logits in descending order
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # Computes the cumulative probabilities using softmax

                    sorted_indices_to_remove = cumulative_probs > top_p  # Computes the indices to remove based on the top-p threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # Shifts the indices to remove
                    sorted_indices_to_remove[..., 0] = 0  # Ensures the first index is not removed

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]  # Retrieves the indices to remove
                    logits[:, indices_to_remove] = -float("inf")  # Sets the logits of the removed indices to negative infinity

                    next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)  # Samples the next token based on the adjusted logits
                    generated = torch.cat((generated, next_token), dim=1)  # Concatenates the generated tensor with the next token

                    if next_token in self.tokenizer.encode(""):  # Checks if the generated text contains an empty token
                        break  # Stops generating further text if an empty token is encountered

                generated_text = self.tokenizer.decode(generated.squeeze().numpy())  # Decodes the generated tensor into text form
                generated_lyrics.append(generated_text)  # Adds the generated lyrics to the list

        return generated_lyrics  # Returns the list of generated lyrics entries
