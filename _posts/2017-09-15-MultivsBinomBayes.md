---
title: Close look @ Sklearn's Naive Bayes algorithms
date: 2017-09-15 14:14:32
categories: naivebayes, sklearn
permalink: /multinbvsbinomnb
layout: taylandefault
published: true
---



# A close look to Sklearn's Naive Bayes (NB) algorithms

## Intro:

Naive Bayes (NB) is a popular algorithm for text-related classification tasks. Sklearn's Multinomial and Bernoulli NB implementations have subtle but quite significant nuances/differences between them. This post will look into these nuances, try to clarify assumptions behind each type of Naive Bayes and try to explain when to use each one.  

`sklearn`'s MultinomialNB and BernoulliNB implementations make certain silent assumptions on what kind of datasets they are being trained on. To use these ML algorithms properly, one needs to prepare the training datasets in accordance with these assumptions, which is often overlooked by ML practitioners and this consequently leads to models that are improperly trained.

---

Let's first see how they behave differently in a very simple example. In the following toy example, our training set is binary, and very small. Let's jump right in:


```python
# First, some imports
import numpy as np
import pandas as pd

X = pd.DataFrame([[0, 1],
              [1, 0],
              [0, 0],
              [0, 0],
              [1, 1]], columns=['x1', 'x2'])
y = np.array([0, 0, 1, 1, 1])


X.join(pd.Series(y, name='y'))
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



So, as you see, we have 5 training samples, with two features. The first two samples belong to class 0, and the other 3 belong to class 1.  

Let's fit a `Multinomial` and `Bernoulli` Naive Bayes classifier to this toy dataset. After the models are fit, let's apply it on a sample and see what happens.


```python
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# no smoothing here, in order to easily understand the computations under the hood
mnb = MultinomialNB(alpha=0)  
bnb = BernoulliNB(alpha=0)
mnb.fit(X, y)
bnb.fit(X, y);
```

The models are fit now. Let's apply it to a sample, where the features `x1` and `x2` are both 1.


```python
new_sample = [[1, 1]]


print(mnb.predict_proba(new))
print(bnb.predict_proba(new))
```

    [[ 0.4  0.6]]
    [[ 0.6  0.4]]


They predict complete opposite probabilities! Ok, what is happening?

Let's compute by hand:

### **Binomial**:

The way we think here is as follows:

Each row is a document.  
Each feature is a flag, representing __if a word exist in that document or not__.
Thus, we end up computing the following conditional probabilities (likelihoods).

* P(<span style="color:green">**a document has x1 in it**</span> \| document is 0) = 1/2  
&nbsp;&nbsp;&nbsp;&nbsp;_Got this from counting the samples where x1 is 1, and class is 0_
* P(<span style="color:green">**a document has x2 in it**</span> \| document is 0) = 1/2  
&nbsp;&nbsp;&nbsp;&nbsp;_Similarly, count samples where x1 is 1, and class is 0_
* P(document is 0) = 2/5  
&nbsp;&nbsp;&nbsp;&nbsp;_Got this from counting 0s in the y array_
    
So, P(new is 0) ~= 1/2.1/2.2/5 = 1/10.  

Similarly,  
    
* P(<span style="color:green">**a document has x1 in it**</span> \| document is 1) = 1/3  
&nbsp;&nbsp;&nbsp;&nbsp;_Got this from counting the samples where x1 is 1, and class is 1_
* P(<span style="color:green">**a document has x2 in it**</span> \| document is 1) = 1/3   
&nbsp;&nbsp;&nbsp;&nbsp;_Similarly, count samples where x1 is 1, and class is 1_
* P(document is 1) = 3/5  
&nbsp;&nbsp;&nbsp;&nbsp;_Got this from counting 1s in the y array_

   
So, P(new is 1) ~= 1/3.1/3.3/5 = 1/15  

So, P(new=0) is 0.6 and P(new=1) = 0.4  (normalized the probas 1/10 and 1/15 here)

---

### **Multinomial**:

The way we think here is different.

Each row is still a document.
Each column is now a count, of a word in the documents. This is why MultinomialNB can work with matrices whose entries are non-binary, but positive.

Thus, we end up computing the following conditional probabilities (likelihoods).

 * P(<span style="color:orange">**given word is x1**</span> \| document is 0) = 1/2  
&nbsp;&nbsp;&nbsp;&nbsp;_This is because documents labeled 0 have only 2 words in them, and 1 of them is x1._
 * P(<span style="color:orange">**given word is x2**</span> \| document is 0) = 1/2  
&nbsp;&nbsp;&nbsp;&nbsp;_Same._
 * P(document is 0) = 2/5  
&nbsp;&nbsp;&nbsp;&nbsp;_Got this from counting 0s in the y array_
 
 So far it is the same, but it differs in below:
 
 * P(<span style="color:orange">**a document has x1 in it**</span> \| document is 1) = 1/2  
&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:red">___Although there are 3 documents labeled 1, there are 2 words in doc1, and only 1 of them is x1___</span>
 * P(<span style="color:orange">**a document has x2 in it**</span> \| document is 1) = 1/2  
&nbsp;&nbsp;&nbsp;&nbsp;_Same._
 * P(document is 0) = 3/5  
&nbsp;&nbsp;&nbsp;&nbsp;_Got this from counting 1s in the y array_
 
Now, work out the probas, and you'll find

P(new=0) is 0.4 and P(new=1) = 0.6.

---

The difference lies in the likelihoods we're computing. In multinomial, we're calculating horizontally (so to speak), the denominator has data from other columns. In binomial, we're calculating vertically only, the denominator does not see any of the other columns.

---

**IMPORTANT**: So with these concepts in mind; it should be clear that using `MultinomialNB` with an input matrix containing a negative number does not make sense. In fact, it will just error out if you try to do so. On the other hand, `BernoulliNB` technically works with continuous data too, because it first __binarizes it (with default threshold 0)__. Trying these algorithms on a general dataset without taking measures against these things is will lead to meaningless models in practice.


# Conclusion:

`sklearn`'s `MultinomialNB` and `BernoulliNB` are implementations of Naive Bayes that are built with NLP-related tasks in mind, especially `MultinomialNB`. Specifically, they assume the input they receive are coming from a text preprocessor such as `CountVectorizer` or `TfidfVectorizer`. One needs to do a lot of cleanup and preparation in order to use it in a more general setting, taking how these algorithms interpret the input data into consideration. In the next post, I will explore a more general implementation of Naive Bayes which will apply outside the NLP related tasks.
