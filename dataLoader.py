import os
import re
import pandas as pd

class LyricsDataLoader:
    """
    Class to load and process lyrics data.
    """
    def __init__(self, data_path):
        """
        Initializes the LyricsDataLoader object.

        Args:
            data_path (str): The base path where the lyrics data is located.
        """
        self.data_path = data_path
        self.album_path = os.path.join(data_path, "Albums")
        self.album_file = os.path.join(data_path, "Albums.csv")
        
    def load_album(self):
        """
        Loads the album data from the CSV file and returns a dataframe.

        Returns:
            pandas.DataFrame: Dataframe with the following columns:
                - ID: Artist ID
                - Albums: Name of the album
        """
        album = pd.read_csv(self.album_file)
        album.drop('Unnamed: 0', axis=1, inplace=True)
        return album
    
    def get_lyrics_dataframe(self):
        """
        Retrieves the lyrics data and returns a dataframe.

        Returns:
            pandas.DataFrame: Dataframe with the following columns:
                - ID: Artist ID
                - Album_path: Path to the album
                - lyrics: Lyrics of the song
        """
        album = self.load_album()
        lyrics_dataframe = pd.DataFrame()
        for i in range(len(album)):
            try:
                tmp_album_path = os.path.join(self.album_path, re.sub('[^a-zA-Z0-9_]', lambda match: '_' if match.group(0) in ('(', ')', ':', '-', '"', "[", "]") else '', album.loc[i]['Albums']))
                tmp_lyrics = [open(os.path.join(tmp_album_path, adir)).read()
                          for adir in os.listdir(tmp_album_path)]
                lyrics_dataframe = pd.concat([lyrics_dataframe, pd.DataFrame({'ID': album.loc[i, "ID"], 'Album_path': tmp_album_path, 'lyrics': tmp_lyrics})])
                lyrics_dataframe.reset_index(drop=True, inplace=True)
            except FileNotFoundError:
                print(f'{album.loc[i]["Albums"]} - File not available in Album directory')
        return lyrics_dataframe

    def clean_lyrics(self, lyrics_dataframe):
        """
        Removes the first part of the lyrics which contains the song name and artist name.

        Args:
            lyrics_dataframe (pandas.DataFrame): Dataframe with lyrics data.

        Returns:
            pandas.DataFrame: Dataframe with cleaned lyrics data.
        """
        lyrics_dataframe['lyrics'] = lyrics_dataframe.lyrics.apply(lambda x: x[x.find('Lyrics'):])
        return lyrics_dataframe
    
    def split_train_test(self, test_set):
        """
        Splits the lyrics dataframe into train and test sets and cleans the lyrics.

        Args:
            test_set (list): List of artist IDs to be included in the test set.

        Returns:
            pandas.DataFrame: Dataframe containing the train set.
            pandas.DataFrame: Dataframe containing the test set.
        """
        lyrics_dataframe = self.get_lyrics_dataframe()
        lyrics_dataframe = self.clean_lyrics(lyrics_dataframe)
        train = lyrics_dataframe[~lyrics_dataframe.ID.isin(test_set)].reset_index(drop=True)
        test = lyrics_dataframe[lyrics_dataframe.ID.isin(test_set)].reset_index(drop=True)
        return train, test
