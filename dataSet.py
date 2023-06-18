import torch
from torch.utils.data import Dataset

class SongLyrics(Dataset):
    """
    Class to load the lyrics data
    """
    def __init__(self, lyrics_df, tokenizer, control_code="", max_length=1024):
        """
        Initializes the SongLyrics dataset.

        Args:
            lyrics_df (pandas.DataFrame): Dataframe containing the lyrics data.
            tokenizer (transformers.Tokenizer): Tokenizer for encoding the lyrics text.
            control_code (str, optional): Control code to prepend to the lyrics text. Default is an empty string.
            max_length (int, optional): Maximum length of the encoded lyrics. Default is 1024.
        """
        self.lyrics_df = lyrics_df
        self.tokenizer = tokenizer
        self.control_code = control_code
        self.max_length = max_length

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.lyrics_df)

    def __getitem__(self, index):
        """
        Retrieves an item from the dataset based on the provided index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            torch.Tensor: Tensor containing the encoded lyrics for the corresponding item.
        """
        lyrics_text = self.lyrics_df.iloc[index]['lyrics']
        encoded_lyrics = self.tokenizer.encode(f"<|{self.control_code}|>{lyrics_text[:self.max_length]}")
        return torch.tensor(encoded_lyrics)
