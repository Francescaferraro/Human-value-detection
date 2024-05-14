This repository contains data and code for the "Human value extraction from arguments: a text generation approach" Master's Degree thesis in Computer Science at the University of Milan.

This work is composed of 2 main steps: the first is about the prompting strategy implemented using the ability of Bert-base-uncased for filling masked texts, and the second is the classification of the original and prompting-expanded datasets using BORN and BERT classifiers. 

Therefore, this repository is divided into 4 directories:
- "Datasets" : contains the original arguments and the three versions of the dataset, i.e. the original and the two versions of the expansions.
- "Prompting strategy" : contains both the .py file and the notebook with the cells outputs for the first part of the work, about the results obtained by the prompting strategy with dictionary associations.
- "Classification" : contains the notebooks for the classification step, done separately with BORN and BERT.
- "Documents" : contains documents that could help to understand the workflow process.
