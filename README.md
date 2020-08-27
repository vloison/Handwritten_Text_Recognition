# Deep Learning framework for Line-level Handwritten Text Recognition

[Presentation of our project](https://docs.google.com/presentation/d/12Z29QPWQubbgZ_PfHG1yqZ3Cal-d6sHWFJcZ0bJVxH8/edit?usp=sharing)

This github provides a framework to train and test CRNN networks on handwritten line-level datasets. \
It wa an internship project under Mathieu Aubry's supervision, at the LIGM lab, located in Paris. 

1. Intro \
    1.a The HTR line-level task \
    1.b The CRNN structure 
    
2. Installation \
    2.a : Install conda environment \
    2.b Download databases
    - IAM dataset
    - ICFHR 2014 dataset

3. How to use
    - Make predictions using our best networks
    - train and test a network on a dataset
    

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
    - The 'lines.txt' file at this link [ADD LINK]. \

- For ICFHR2014 dataset, you need to download the 'BenthamDatasetR0-GT' folder at this [link](https://zenodo.org/record/44519#.X0eXbHkzY2x). \ 

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

### 3.a Make predictions on your own dataset
Coming soon...

### 3.b Train a network from scratch

``` 
python train.py --dataset dataset  -- data_dir data_dir --logdir logdir
``` 
Before running the code, make sure that you change `ROOT_PATH` variable at the beginning of `params.py` to the path of the folder you want to save your models in. 
Main arguments : 
- `--dataset`: name of the dataset to train and test on. 
Supported values are `ICFHR2014` and `IAM`.
- `--data_path`: location of the train dataset folder on local machine. See section [??] for downloading datasets.


Main learning arguments : 
- `data_aug`: If set to `True`, will apply random affine data transformation to the training images.
- `--optimizer`: Which optimizer to use. 
Supported values are `rmsprop`, `adam`, `adadelta`, and `sgd`. 
We recommend using RMSprop, which got best results in our experiments. See `params.py` for optimizer-specific parameters.

- `--epochs` : Number of training epochs
- `--lr`: Learning rate at the beginning of training.
- `--milestones`: List of the epochs at which the learning rate will be divided by 10. 

## References
Graves et al. [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://mediatum.ub.tum.de/doc/1292048/file.pdf) \
Sánchez et al. [A set of benchmarks for Handwritten Text Recognition on historical documents](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302006) \
Dutta et al. [Improving CNN-RNN Hybrid Networks for
Handwriting Recognition](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2018/improving-cnn-rnn.pdf)

U.-V. Marti, H. Bunke [The IAM-database: an English sentence database for offline handwriting recognition](https://link.springer.com/article/10.1007/s100320200071)

https://github.com/Holmeyoung/crnn-pytorch \
https://github.com/georgeretsi/HTR-ctc \
Synthetic line generator : https://github.com/monniert/docExtractor (see [paper](http://imagine.enpc.fr/~monniert/docExtractor/docExtractor.pdf) for more information)