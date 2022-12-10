# Applied Deep Learning - W2022

## Assigment 1 - Initiate

### 1. Scientific papers related to the topic

Rodríguez, Á. I., & Iglesias, L. L. (2019). Fake news detection using deep learning. arXiv preprint arXiv:1910.03496.
https://arxiv.org/abs/1910.03496

S. H. Kong, L. M. Tan, K. H. Gan and N. H. Samsudin, "Fake News Detection using Deep Learning," 2020 IEEE 10th Symposium on Computer Applications & Industrial Electronics (ISCAIE), 2020, pp. 102-107, doi: 10.1109/ISCAIE47305.2020.9108841.
https://ieeexplore.ieee.org/abstract/document/9108841

S. Rastogi and D. Bansal, "Time is Important in Fake News Detection: a short review," 2021 International Conference on Computational Science and Computational Intelligence (CSCI), 2021, pp. 1441-1443, doi: 10.1109/CSCI54926.2021.00286.
https://ieeexplore.ieee.org/document/9799153

H. Matsumoto, S. Yoshida and M. Muneyasu, "Propagation-Based Fake News Detection Using Graph Neural Networks with Transformer," 2021 IEEE 10th Global Conference on Consumer Electronics (GCCE), 2021, pp. 19-20, doi: 10.1109/GCCE53005.2021.9621803.
https://ieeexplore.ieee.org/document/9621803

S. H and S. B, "A Review on News-Content Based Fake News Detection Approaches," 2022 International Conference on Computing, Communication, Security and Intelligent Systems (IC3SIS), 2022, pp. 1-6, doi: 10.1109/IC3SIS54991.2022.9885447.
https://ieeexplore.ieee.org/document/9885447

### 2. Decision topic:
The decided topic is in the field of Natural Language Processing, focusing on FakeNews Detection.

### 3. Type of Project:
Bring your own method - build or re-implement a NN that operates on an existing dataset that is already publicy available.

### 4. Summary
#### a. Project description

In summary, the project aims to create a classification model to detect fake news. Since the type of project chosen was "Bring Your Own Method", an attempt will be made to improve the results obtained using BERT, the state-of-the-art approach as stated in the article by RODRÍGUEZ and IGLESIAS (2019). The project will consist of preparing the datasets, researching and experimenting with different approaches, and comparing the metrics presented in the article with the results obtained. At the end, a report and presentation presenting an introduction, methodology, results and conclusions will be made.

#### b. Dataset description

For a first approach, two datasets are chosen as the same datasets indicated by RODRÍGUEZ and IGLESIAS (2019)[2], one containing fakenews and the other, the real news. Since they come from different sources, they will have to be modified to achieve same format and merged to fit on the model. Additionally, since the real news dataset is huge in size and comprisse several folders for different years, it was chosen only the year 2016 since it is the same year as the fakenews dataset.

* fake news: BS Detector dataset: https://www.kaggle.com/datasets/mrisdal/fake-news size: 12999 rows
The fake news were news collected from 244 different website that were classified as fake by the "BS Detector Chrome Extension by Daniel Sieradsk".

* real news: New York Times dataset (year 2016):https://www.kaggle.com/datasets/tumanovalexander/nyt-articles-data size: 105606 rows

As an second approach, new datasets were added to train the model and evaluate how it would perfom with more diverse data, from diverent sources.

* ISOT dataset[1]: A dataset that has combined fake news and real news. The real news are articles from Reuters.com (news website) and the fake news are articles collected from unreliable websites that were flagged by Politifact (a fact-checking organization in the USA) and Wikipedia. Source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download
Details: https://www.uvic.ca/engineering/ece/isot/assets/docs/ISOT_Fake_News_Dataset_ReadMe.pdf




#### c. Work-breakdown structure with time estimates:

* Research (reading papers, understand models, etc): 10 hours
* Merge datasets, preprocessing: 4 hours
* Design and building appropriate network: 5 hours
* Training and fine-tuning: 4 hours
* Building application to present the results: 5 hours
* Writing final report: 6 hours
* Presentation: 4 hours


## Assigment 2 - Hacking

### Methodology
First approach was to implement BERT, as specified bevor. Due to it be very computationally expensive, Google Colab was used to run the code. Even using Google Colab GPU resource, it took around 5 hours and the results were not satisfatory (all the predictios were as 1). 
Afterwards, it was tried to implement LSTM model as it is faster and an well-estabilished model. It was again implemented in Google Colab, it took around 1 minute to train. It reached on the first run an accuracy of 0.98, without optimizing parameters. 

Since the accuracy was very high on the first try, it could be an indication of overfit. On this way, it was tried to modify the datasets and parameters to see how the results vary.

As part of the preprocessing step, stopwords were removed, as well as cracters not identified as words ('!"#$%&()*+,-./:;<=>?@[|}~)

. To avoid overfitting, the training epochs are stopped as soon as the validation accuracy does not improve after 3 iterations.

The model were structure as following:

![image](https://user-images.githubusercontent.com/47119194/206860189-0f966356-1b6d-4564-b252-65e7c37c7901.png)



Approaches tried:

Baseline parameters:
    max_words = 1000
    max_len = 300
    n_batchsize = 128
    n_epochs = 20
    dropout = 0.2

* Baseline: 2 datasets: fakenews from BS Detector and real news from New York Times (details explained on the previous section). Number of words = 1000 and lenght of sentences = 300
 > f1score: 0.97 for both classes
 
* Increase of number of words and lenght of setences with same datasets: Number of words = 5000 and lenght of sentences = 500. 
> f1socre: 0.97 for class 0 and 0.99 for class 1

* Lower number of words and lenght of sentences with same datasets:  Number of words = 300 and lenght of sentences = 50.
> f1score: 0.97 for both classes

* Set datasets of fake and real as similar size (15000 rows for real news and 12999 for fake news), using baseline parameters.
> f1socre: 0.97 for class 0 and 0.98 for class 1

* Add two more datasets from ISOT Dataset[1], use baseline parameters.
> f1socre: 0.96 for class 0 and 0.99 for class 1

The final two approaches were done with the same parameters as the baseline approach since the evaluation on validation set did not presentedn any significance change.

Lastly, the model from the last approach (4 datasets and baseline parameters) was chosen as the final model, since it present higher number of datasets what makes it amore complete model and performed as good as the others on validation set. Therefore it was finally tested on the test set. 

Results on test set: f1socre: 0.96 for class 0 and 0.99 for class 1


### Time required

    merging dataset: 1,5 hours
    BERT: 5 hours
    LSTM: 4 hours
    add stopwords: 1 hour
    create pipeline: 3 hours
    test different datasets: 2 hours
    test different parameters: 2 hours
    ask feedback from colleague: 1 hour
    add earlystop and fix small errors: 1 hour
    write report for part 2: 1 hour
    
    
## References:

[1] Ahmed H, Traore I, Saad S. “Detecting opinion spams and fake news using text classification”, Journal of Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018

[2] Rodríguez, Á. I., & Iglesias, L. L. (2019). Fake news detection using deep learning. arXiv preprint arXiv:1910.03496.
