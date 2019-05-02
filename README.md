# A Deep Learning Based Morph Analyzer

This repository holds the code for a Neural Network based Morph Analyzer built for Hindi Language. The analyzer accepts a UTF-8 encoded sentence in Devnagri Script and outputs a list of words complete with its lemma and a set of 6 Morphological features namely POS, Gender, Person, Case, Number, TAM Marker.

The analyzer employs a CNN-RNN model with multi-task learning to jointly learn all the six morphological tags and the lemma for each word. 

## Installation

Create a python3 virtual environment with keras, scikit-learn, pandas, numpy and matplotlib installed.
 
Clone the repository and you are ready to go.

# Downloading Datasets

The Hindi-Urdu Dependency Treebanks can be download from [this](http://ltrc.iiit.ac.in/hutb_release/) webpage hosted by IIIT-Hyderabad.

The downloaded datasets should can then be extracted in the `datasets` directory.

## Usage

The code for generating predictions rests in the make_prection.py file. It takes input from input.txt and prints the output to output.txt. Should you need to predict more than one sentence, just separate the sentences by newline.

```bash
python make_prediction.py
```

### Creating Encoders

The file make_encoders.py hosts the code to generate Label Encoders for each of the Morphological Tags. The encoders are pickled and saved to the file tag_encoders.pickle
