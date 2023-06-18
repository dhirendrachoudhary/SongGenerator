# Song Generator

This project is a song generator that utilizes a pre-trained language model to generate lyrics in the style of a given artist, in this case, Taylor Swift.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Future Plans](#future-plans)

## Introduction

The Song Generator is designed to generate song lyrics in the style of Taylor Swift. It uses a pre-trained language model to generate lyrics based on a given prompt.

## Installation

To use the Song Generator, follow these steps:

1. clone the repository:<br>
    `git clone https://github.com/dhirendrachoudhary/RnD_AI_Challenge.git`
   
3. Create a virtual environment and activate it:<br>
    `python3 -m venv env`
    `source env/bin/activate`

4. Install the dependencies:<br>
    `pip install -r requirements.txt`

5. Download data and model<br>
    Download the data in the root directory<br>
    `wget https://drive.google.com/drive/folders/1LWe1VgEaTr5Bk5tJc81v4bgfqZGn7xMY?usp=sharing`

    Download the model in the root directory<br>
    `wget https://drive.google.com/drive/folders/1ZLSaiyk7sZpTqAvs23kV-Gqh6sbFVqkf?usp=sharing`

6. Run the code:<br>
    `python songGenerator.py`

## Usage

1. Prepare the dataset:

    To train the model, you need a dataset of Taylor Swift's song lyrics. Place the text files containing the lyrics in the `data` directory.

2. Train the model:

    Run the `songGenerator.py` script with `train=True` to train the song generator model. This will load the dataset, preprocess the lyrics, and train the model. Adjust the hyperparameters in the script as desired.

3. Generate lyrics:

    After training the model, you can generate lyrics using the `songGenerator.py` script but make `train=False` and give text prompt as input. This will load the pre-trained model and generate lyrics based on the given prompt.

4. Explore the code:

    The codebase includes several modules and classes that handle data loading, model architecture, training, and generation. Refer to the code files for more details on each component's functionality and implementation.

## Technical Details

The Song Generator is built using the following technologies and libraries:

- Python
- PyTorch
- Transformers library

The key components of the codebase include:

- `LyricsDataLoader`: Loads the dataset of Taylor Swift's song lyrics.
- `SongLyrics`: Custom dataset class for tokenizing and encoding the lyrics.
- `LyricsGenerator`: Generates lyrics using the pre-trained language model.
- `LyricsModel`: Wrapper class for the GPT-2 language model.
- `songGenerator`: Main function to train or generate lyrics based on user input.

## Limitations

- The performance of the song generator heavily depends on the quality and diversity of the training dataset. If the dataset is limited or biased, the generated lyrics may lack variety and creativity.
- The generated lyrics are purely based on statistical patterns learned by the language model and may not always exhibit coherent storytelling or meaningful composition.
- The model's performance and generation quality may vary based on the size of the dataset, model architecture, and hyperparameter settings.

## Future Plans

In future iterations of the project, the following improvements and enhancements can be considered:

- Improving the model's performance and generating more coherent and creative lyrics through advanced training techniques such as fine-tuning, transfer learning, or custom architectures.
- Enhancing the user interface and interactivity of the song generator, allowing users to customize the style, theme, or mood 
- Expanding the dataset to include lyrics from a broader range of artists and genres to generate diverse lyrics.
- Integrating additional components such as melody generation or album cover generation to create a more comprehensive song creation system.

Due to time constraints, the implementation of the album cover generation feature was not completed in this version of the project.
