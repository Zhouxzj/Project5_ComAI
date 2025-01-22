# Description

This is the repository of my Final Project of Contemporary AI

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==2.5.1+cu124

- transformers==4.25.1  # Based on the usage of BertTokenizer and BertModel

- Pillow==11.0.0  # PIL is Pillow

- scikit-learn==1.2.0  # Adjust the version if needed based on your setup

- matplotlib==3.9.2

- torchvision==0.20.1+cu124

- pandas==2.2.3

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description. You need to put project5 data in the current directory and name it as dataset. You need to download bert_uncased from this link: https://pan.baidu.com/s/1cwXipaW2_coEC-lG5NGhKw?pwd=4jk8 and the key is 4jk8

```python
|-- dataset # the datasets for the lab
    |-- data/ # images and txts
    |-- test_without_lab.txt # prediction document
    |-- train.txt # data for train
|-- bert_uncased # bert_uncased model
|-- train.py # use to construct and train my model
|-- project5.ipynb # jupyter notebook version of train.py, saving some plots and outputs of the training process
```

## Run pipeline
1. Entering the Final_project directory. Notice, you should rename the datasets and place them in the right directory.
```python
python train.py
```


## Attribution

Parts of this code are based on the following repositories:

- [BERT](https://github.com/google-research/bert?tab=readme-ov-file)
