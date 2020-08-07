import nltk
import os
import torch as torch

# ------------------------------------------------
"""
In this block : metrics
    CER
    WER
    """


def CER(label, prediction):
    return nltk.edit_distance(label, prediction)/len(label)


def WER(label, prediction):
    return nltk.edit_distance(prediction.split(' '), label.split(' ')) / len(label.split(' '))


if __name__ == "__main__":
    w1 = "Trouver le code"
    w2 = "Trouver des code"
    print(WER(w1, w2))

