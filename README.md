# Word Segmentation Experiments

This repository contains the code to generate an artificial language like that used by 
[Frank et al. (2013)](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0052500&type=printable).
Specifially, this language is constructed from 336 CV syllables composed of 24 consonants and 14 vowels, 
which are concatenated into words with lengths drawn from a Poisson distribution with mean 2 and adding 1 to 
avoid zero-length words. 
The words are given a Zipfian frequency distribution and sampled according to this distribution to create 
sentences, where sentence length is also drawn from a Poisson distribution with mean 2 and 2 is added to avoid 
sentences of length 1. 
For the experiment, we create 1000 unique words and sample from these to create sentences until 60k tokens have been 
generated for our training data; for our testing data we sample until 400 tokens have been generated. 
This sampling is conducted 10 times and model results are averaged over these 10 runs. 
To generate the artificial language data, run `python3 language.py`

We also include the code to run the basic Transitional Probability segmentation model on this data and report
the average and standard deviation of the results. 
To do so, you will need to install the [wordseg](https://docs.cognitive-ml.fr/wordseg/installation.html) package;
then run `python3 tp.py` to run the basic TP model over syllables on this data and report averaged results. 


