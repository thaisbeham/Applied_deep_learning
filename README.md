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

The datasets used are the same as indicated by RODRÍGUEZ and IGLESIAS (2019). Two different datasets are required, one containing fakenews and the other, the real news. Since they come from different sources, they will have to be modified to achieve same format and merged to fit on the model.

For fake news: https://www.kaggle.com/datasets/mrisdal/fake-news 

For real news: https://www.kaggle.com/datasets/tumanovalexander/nyt-articles-data

#### c. Work-breakdown structure with time estimates:

* Research (reading papers, understand models, etc): 10
* Merge datasets, preprocessing: 4
* Design and building appropriate network: 5
* Training and fine-tuning: 4
* Building application to present the results: 5
* Writing final report: 6
* Presentation: 4


## Assigment 2 - Hacking

### Methodology
First approach was to implement BERT, as specified bevor. Due to it be very computationally expensive, Google Colab was used to run the code. Even there it took around 5 hours and the results were not satisfatory, since all the predictios were as 1. 
Afterwards it was tried to implement LSTM model as it is faster and an well-estabilished  model. It was again implemented in Google Colab, taking less than 1 hour. It reached on the first run an accuracy of 0.98, without optimizing parameters. 

Since the accuracy was very high on the first try, it could be an indication of overfit. On this way, it was tried to modify the datasets to see how the results vary.

* Approach 1: LSTM using the 13000 real news (from year 2016) and 13000 fake news from from the sources indicated on the dataset section

### Approach 1 results:


122/122 [==============================] - 1s 6ms/step - loss: 0.1099 - accuracy: 0.9608
Test set
  Loss: 0.110
  Accuracy: 0.961
122/122 [==============================] - 0s 4ms/step
Number Real news: 1988
Number Fake news: 1912

Classification Report

              precision    recall  f1-score   support

     class 0       0.95      0.97      0.96      1891
     class 1       0.97      0.96      0.96      2009

    accuracy                            0.96      3900
    macro avg       0.96      0.96      0.96      3900
    weighted avg    0.96      0.96      0.96      3900


Confusion Matrix

              class_0 True  class_1 True
    class_0 pred    1825            66
    class_1 pred      87          1922



Approach 2: using all the data of real news

Classification Report

              precision    recall  f1-score   support

     class 0       0.98      0.93      0.96      1881
     class 1       0.99      1.00      0.99     10371

    accuracy                           0.99     12252
    macro avg     0.98      0.96       0.97     12252
    weighted avg  0.99      0.99       0.99     12252


Confusion Matrix

                  class_0 True  class_1 True
    class_0 pred     1751           130
    class_1 pred      32           10339
    
    
### Aprroach 3: Using 1 more dataset for fake news and one more for real news: 
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download


Number Real news: 5773
Number Fake news: 5162

Classification Report

              precision    recall  f1-score   support

     class 0       0.97      0.93      0.95      5406
     class 1       0.93      0.97      0.95      5529

    accuracy                           0.95     10935
    macro avg      0.95      0.95      0.95     10935
    weighted avg   0.95      0.95      0.95     10935


Confusion Matrix

                    class_0 True  class_1 True
    class_0 pred          5013           393
    class_1 pred           149          5380
