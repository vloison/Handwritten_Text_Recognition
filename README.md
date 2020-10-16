# Deep Learning framework for Line-level Handwritten Text Recognition

[Short presentation of our project](https://docs.google.com/presentation/d/12Z29QPWQubbgZ_PfHG1yqZ3Cal-d6sHWFJcZ0bJVxH8/edit?usp=sharing)

1. [Introduction](#1-introduction) 
    
2. [Installation](#2-installation) \
    2.a [Install conda environment](#2a-download--and-activate-conda-environment) \
    2.b [Download databases](#2b-download-databases)
    - IAM dataset
    - ICFHR 2014 dataset

3. [How to use](#3-how-to-use) \
    3.a [Make predictions on unlabelled data using our best networks](#3a-make-predictions-on-your-own-unlabelled-dataset) \
    3.b [Train and test a network from scratch](#3b-train-a-network-from-scratch)\
    3.c [Test a model without retraining it](#3c-test-a-model-without-retraining-it)

4. [References](#4-references)

5. [Contact](#5-contact)


## 1. Introduction 

This work was an internship project under Mathieu Aubry's supervision, at the LIGM lab, located in Paris. 

In HTR, the task is to predict a transcript from an image of a handwritten text. A commonly used structure for this task is Convolutional Recurrent Neural Networks (CRNN). 
One CRNN network consists of a feature extractor (often with convolutional layers), followed by a recurrent network (LSTM).

**This github provides a framework to train and test CRNN networks on handwritten grayscale line-level datasets.**
**This github also provides code to generate predictions on an unlabelled, line-level, grayscale line-level dataset.**
There are several options for the structure of the CRNN used, image preprocessing, dataset used, data augmentation. 

![alt text][image]

[image]: https://github.com/vloison/Handwritten_Text_Recognition/blob/master/crnn_image.PNG



## 2. Installation

### Prerequisites

Make sure you have Anaconda installed (version >= to 4.7.10, you may not be able to install correct dependencies if older). 
If not, follow the installation instructions provided at 
https://docs.anaconda.com/anaconda/install/.

Also pull the git. 


### 2.a Download  and activate conda environment
Once in the git folder on your machine, run the command lines :
``` 
conda env create -f HTR_environment.yml
conda activate HTR 
```   

### 2.b Download databases
You will only need to download these databases if you want to train your own network from scratch. The framework is built to train a network on one of these 2 datasets : 
[IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) and [ICFHR2014 HTR competition](http://www.transcriptorium.eu/~htrcontest/contestICFHR2014/public_html/). [ADD REF TO SLIDES]
- Before downloading IAM dataset, you need to register on [this](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) website. Once that's done, you need to download : 
    - The 'lines' folder at this [link](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database/download-the-iam-handwriting-database).
    - The 'split' folder at this [link](http://www.fki.inf.unibe.ch/DBs/iamDB/tasks/largeWriterIndependentTextLineRecognitionTask.zip).
    - The 'lines.txt' file at this [link](http://www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/lines.txt).

- For ICFHR2014 dataset, you need to download the 'BenthamDatasetR0-GT' folder at this [link](https://zenodo.org/record/44519#.X0eXbHkzY2x).

Make sure to download the two databases in the same folder. Structure must be 
```
Your data folder / 
    IAM/
        lines.txt
        lines/
        split/
            trainset.txt
            testset.txt
            validationset1.txt
            validationset2.txt
            
    ICFHR2014/
        BenthamDatasetR0-GT/ 

    Your own dataset/
```


## 3. How to use

### 3.a Make predictions on your own unlabelled dataset

Running this code will use model stored at `model_path` to make predictions on images stored in `data_path`.
The predictions will be stored in `predictions.txt`  in `data_path` folder.

``` 
python lines_predictor.py --data_path datapath  --model_path ./trained_networks/IAM_model_imgH64.pth --imgH 64
``` 
/!\ Make sure that each image in the data folder has a unique file name. When you use our trained model with imgH as 64 (i.e. IAM_model_imgH64.pth), you have to set the argument --imgH as 64.
### 3.b Train a network from scratch

``` 
python train.py --dataset dataset  --tr_data_path data_dir --save_model_path path
``` 
Before running the code, make sure that you change `ROOT_PATH` variable at the beginning of `params.py` to the path of the folder you want to save your models in. 
Main arguments : 
- `--dataset`: name of the dataset to train and test on. 
Supported values are `ICFHR2014` and `IAM`.
- `--tr_data_path`: location of the train dataset folder on local machine. See section [??] for downloading datasets.
- `--save_model_path`: path of the folder where model will be saved if `params.save` is set to True.

Main learning arguments : 
- `--data_aug`: If set to `True`, will apply random affine data transformation to the training images.
- `--optimizer`: Which optimizer to use. 
Supported values are `rmsprop`, `adam`, `adadelta`, and `sgd`. 
We recommend using RMSprop, which got best results in our experiments. See `params.py` for optimizer-specific parameters.

- `--epochs` : Number of training epochs
- `--lr`: Learning rate at the beginning of training.
- `--milestones`: List of the epochs at which the learning rate will be divided by 10. 

- `feat_extractor`: Structure to use for the feature extractor. Supported values are `resnet18`, `custom_resnet`, and `conv`.
    - `resnet18` : standard structure of resnet18. 
    - `custom_resnet`: variant of resnet18 that we tuned for our experiments. 
    - `conv`: Use this option if you want to use a purely convolutional feature extractor and not a residual one. 
    See conv parameters in `params.py` to choose conv structure.

### 3.c Test a model without retraining it
Running this code will compute the average CER and WER of model stored at `pretrained_model` path on the testing set of chosen `dataset`.
```
python train.py --train '' --save '' --pretrained_model model_path --dataset dataset --tr_data_path data_path 
```

Main arguments : 
- `--pretrained_model`: path to state_dict of pretrained model. 
- `--dataset`: Which dataset to test on. 
Supported values are `ICFHR2014` and `IAM`.
- `--tr_data_path`: path to the dataset folder (see section [??])
## 4. References
Graves et al. [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://mediatum.ub.tum.de/doc/1292048/file.pdf) \
Sánchez et al. [A set of benchmarks for Handwritten Text Recognition on historical documents](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302006) \
Dutta et al. [Improving CNN-RNN Hybrid Networks for
Handwriting Recognition](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/improving-cnn-rnn.pdf)

U.-V. Marti, H. Bunke [The IAM-database: an English sentence database for offline handwriting recognition](https://link.springer.com/article/10.1007/s100320200071)

https://github.com/Holmeyoung/crnn-pytorch \
https://github.com/georgeretsi/HTR-ctc \
Synthetic line generator : https://github.com/monniert/docExtractor (see [paper](http://imagine.enpc.fr/~monniert/docExtractor/docExtractor.pdf) for more information)


## 5. Contact
If you have questions or remarks about this project, please email us at [virginie.loison@eleves.enpc.fr]() and [xiwei.hu@telecom-paris.fr]().
