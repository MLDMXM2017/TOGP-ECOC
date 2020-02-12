# TOGP-ECOC

Ternary Operator Based Genetic Programming - Error Correcting Output Codes

This is the implementation for paper: [A Ternary Operator Based Genetic Programming Approach to Improving Error Correcting Output Codes]

## Acknowledgement

- Codes about Genetic Programming is modified from  [Pyevolve](<https://github.com/perone/Pyevolve>)

- Codes about Output-Code-Classifier is modified from  [scikit-learn 0.18](<https://github.com/scikit-learn/scikit-learn/tree/0.18.X>)

## Environment

- **Windows 10 64 bit** 

- **python 2.7**

- **Excel**

  Enable the macro in excel, so it can extract result from file automatically.

- **scikit-learn 0.18**

  Anaconda is strongly recommended, run the following command in the Powershell, all necessary python packages for this project will be installed:

  ```shell
  conda install scikit-learn==0.18
  ```


## Dataset

- **Data format**

  Raw data is put into the folder ```($root_path)/data/raw/```.
  Normalized data is put into the folder ```($root_path)/data/normalized/```. You can use ```($root_path)/preprocess/Normalization.py``` to get the normalized data.
  Each dataset should be divided into ```dataname_train.data```, ```dataname_test.data``` and ```dataname_validation.data``` with ratio of 2:1:1, and stored in ```($root_path)/data/split/```. You can use ```($root_path)/preprocess/DataSpliter.py``` to get the split data.
  In the sub-dataset, each column is a sample, the first line represents the labels, the rest are feature space. 
  Please note that invalid sample, such as value missed, will cause errors.
  
  There are two micro-array datasets in the folder ```($root_path)/data/``` as an example. 
  Dataset information:
    name: Leukemia1, Leukemia2
    class num: 3
    feature num: 29
    sample num: 72

- **Data processing**

  Feature Selection and Scaling will be done automatically. 


## Multi-processing Mode

- **Config**

  Firstly, make configuration in ```Configurations.py```. In Multi-processing mode, you need not to pay attention to 'dataName' and 'aimFolder'.
  
- **Run the following command**

  It will traversal all datasets given by the main function in ```_ParallelRunner.py```, each dataset will be run for 10 times.

  ```python
  python _ParallelRunner.py
  ```

- **Analyze result**

  All result infos will be written into the folder. For example, if you set version = "4.0", result infos will be found in ```($root_path)/Results/4.0/```

- **Local optimization analysis result**
  After running ```_ParallelRunner.py```, run ```($root_path)/postprocess/localImproveAnalysis.py```, and you can get the results before and after local optimization in the folder ```($root_path)/Results/4.0/a_s/Analysis/localImprovement/```.

- **Operators statistics result**
  After running ```_ParallelRunner.py```, run ```($root_path)/postprocess/operatorstatistic.py```, and you can get the results before and after local optimization in the folder ```($root_path)/Results/4.0/a_s/Analysis/operators/```.

## Single-processing Mode

  This is useful when you want to debug.

- **Config**

  Make configuration in ```Configurations.py```. In Single-processing mode, 'dataName' and 'aimFolder' should also be set.
  
- **Run the following command**

  It will do training and testing on the dataset given in the ```Configurations.py```.

  ```python
  python Main.py
  ```

- **Parse and analyze result**

   In this Mode, part of the result will be printed on the terminal. You can find all result in the Results folder.
   But, there will be no automatic analyzing. After running _ParallelRunner.py, you can open the rTemplate.xlsm in the folder ($root_path)/Results/a_s4.0/ and click the button "read data" to get all the results.

## Attention

Make sure the version folder is not exit every time you run it. Or errors will happen in the following way:
- In Single-processing Mode, the result in the terminal is right, but the result written ino the file might be wrong.
- In Multi-processing Mode, the result could not be read and parsed correctly.

A suggestion is to change the 'version' in the ```Configurations.py``` every time you run it.

