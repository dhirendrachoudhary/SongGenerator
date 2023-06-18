from model import LyricsModel
from dataSet import SongLyrics
from dataLoader import LyricsDataLoader
import nltk
nltk.download('punkt')

def train_or_generate(train=False, train_data=None, test_data=None):
    """
    Trains the model or generates lyrics based on the prompt.

    Args:
        train (bool, optional): Specifies whether to train the model. Default is False.
        train_data (str, optional): The path to the training data. Required if train is True.
        test_data (str, optional): The prompt text for generating lyrics. Required if train is False.

    Returns:
        str: The generated lyrics.
    """
    if train:
        # Create the LyricsModel object
        lyrics_model = LyricsModel()
        
        # Load and preprocess the train data
        train_dataset = SongLyrics(train_data, lyrics_model.tokenizer, control_code="")

        # Train the model
        lyrics_model.train(train_dataset, batch_size=16, epochs=20)

        # Save the trained model
        lyrics_model.save_model("model/model.pt")
    else:
        # Load the saved model
        lyrics_model = LyricsModel()
        lyrics_model.load_model("model/model.pt")

        # Generate lyrics
        generated_lyrics = lyrics_model.generate(test_data, entry_count=2)

        return generated_lyrics[0]


def calculate_bleu_score(generated_text, reference_text):
    """
    Calculates the BLEU score between generated text and reference text.

    Args:
        generated_text (str): The generated lyrics text.
        reference_text (str): The reference lyrics text.

    Returns:
        float: The BLEU score.
    """
    reference_corpus = [nltk.word_tokenize(reference_text)]
    generated_corpus = nltk.word_tokenize(generated_text)
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference_corpus, generated_corpus)
    return bleu_score


if __name__ == "__main__":
    # Train the model
    DATA_PATH = 'data/'
    test_set = [1027134, 1013719, 1013715, 1013718, 1008313, 969188, 962334, 959034, 949856, 945000]

    loader = LyricsDataLoader(DATA_PATH)
    train, test = loader.split_train_test(test_set)

    input_text = test['lyrics'][10][:300]
    generated_text = train_or_generate(train=False, train_data=train, test_data=input_text)
    print(f"\nPrompt text - {input_text}", end='\n\n')
    print(f"\nGenerated text  - {generated_text}")

    print(f"bleu score - {calculate_bleu_score(generated_text, input_text)}")
