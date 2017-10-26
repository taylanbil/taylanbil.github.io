---
title: Boosting the Naive Bayes Classifier
date: 2017-10-26 12:42:20
categories: naivebayes
permalink: boostedNB
layout: taylandefault
published: true
---



# Gradient Boosting the Naive Bayes algorithm

## 0. Intro

In the [last post](/vanillanb), I have included a general implementation of the Naive Bayes algorithm. This implementation differed from `sklearn`'s Naive Bayes algorithms as discussed [here](/multinbvsbinomnb).  

It is well known that Naive Bayes does well with text related classification tasks. However, it is not routinely successful in other areas. In this post, I will try to boost the Naive Bayes algorithm in order to end up with a stronger algorithm that does well more often and in general.  

I organized this post as follows:

* First part will import the code from a file named [nb.py](https://github.com/taylanbil/naivebayes/blob/master/nb.py) and it will show the usage.
* Second part will try the code on three publicly available datasets that can be found in the UCI public ML dataset archive.
* Last part will discuss the implementation by diving deeper into the code.

Before we get started, let's have a quick refresher about the idea behind boosting; very roughly, it is the idea to stack the so called weak-classifiers on top of each other, in such a way that the next one learns from the mistakes of the previous ones combined. In this notebook, our weak-learners are Naive Bayes classifiers. Traditionally though, weak learners are decision-stumps, or in other words very shallow decision trees.

Disclaimer: Just as in the previous posts, the goal is to get to a working implementation, so the code is sub-optimal.  
Let's dive right into it...

---

## 1. Intro: Baby steps

Let's try the boosted NB on the simple, toy "play tennis" dataset. This is the same dataset as in the previous [NB post](/vanillanb).


```python
import pandas as pd

# # Data is from below. Hardcoding it in order to remove dependency
# data = pd.read_csv(
#     'https://raw.githubusercontent.com/petehunt/c4.5-compiler/master/example/tennis.csv',
#     usecols=['outlook', 'temp', 'humidity', 'wind', 'play']
# )

data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No'],
]
data = pd.DataFrame(data, columns='Outlook,Temperature,Humidity,Wind,PlayTennis'.split(','))
X = data[data.columns[:-1]]
y = data.PlayTennis == 'Yes'
data
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
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>PlayTennis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sunny</td>
      <td>Hot</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Rain</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Overcast</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>High</td>
      <td>Weak</td>
      <td>No</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sunny</td>
      <td>Cool</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sunny</td>
      <td>Mild</td>
      <td>Normal</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Overcast</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Overcast</td>
      <td>Hot</td>
      <td>Normal</td>
      <td>Weak</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rain</td>
      <td>Mild</td>
      <td>High</td>
      <td>Strong</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>



So we have 4 predictor fields, 14 samples and a binary classification problem. Let's get the boosted NB classifier from [nb.py](https://github.com/taylanbil/naivebayes/blob/master/nb.py) and try it out. The class is called `NaiveBayesBoostingClassifier`. The api is again very `sklearn`-like, with methods such as `fit`, `predict`, `predict_proba`, `decision_function` etc. Let's also compare the boosted NB with the vanilla NB, `NaiveBayesClassifier`, which is the same as before.

We will have 2 versions of the boosted NB.

1. First one with only 1 iteration, that is, 0 boosting iterations. This case is therefore equivalent to vanilla NB.
1. Second one with 2 iterations. The first vanilla NB iteration, and 1 boosting iteration on top of that.

The goal is to observe the difference in the calculated posterior probabilities. We will also include the `NaiveBayesClassifier`, and observe that it produces the same posteriors as the trivial boosting case.


```python
%run nb.py

nb = NaiveBayesClassifier()
# 1 boosting iteration means, no boosting, and this should spit out exactly the same probas.
fake_boosting = NaiveBayesBoostingClassifier(n_iter=1)  
# the following is 2 boosting iterations, so the posteriors should differ a bit;
real_boosting = NaiveBayesBoostingClassifier(n_iter=2)
fake_boosting.fit(X, y)
real_boosting.fit(X, y)
nb.fit(X, y)

out = pd.Series(nb.predict_proba(X)[1], name='vanilla')
out = pd.DataFrame(out)
cols = ['vanilla', 'boost_fake', 'boost_once', 'truth']
out['boost_once'] = real_boosting.decision_function(X)
out['boost_fake'] = fake_boosting.decision_function(X)
out['truth'] = y
out = out[cols]
out
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
      <th>vanilla</th>
      <th>boost_fake</th>
      <th>boost_once</th>
      <th>truth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.312031</td>
      <td>0.312031</td>
      <td>0.313928</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.162746</td>
      <td>0.162746</td>
      <td>0.164083</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.751472</td>
      <td>0.751472</td>
      <td>0.752014</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.573354</td>
      <td>0.573354</td>
      <td>0.573302</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.875858</td>
      <td>0.875858</td>
      <td>0.870270</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.751472</td>
      <td>0.751472</td>
      <td>0.745584</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.918955</td>
      <td>0.918955</td>
      <td>0.915018</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.430499</td>
      <td>0.430499</td>
      <td>0.432647</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.798736</td>
      <td>0.798736</td>
      <td>0.794659</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.854638</td>
      <td>0.854638</td>
      <td>0.851980</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.586325</td>
      <td>0.586325</td>
      <td>0.584993</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.683522</td>
      <td>0.683522</td>
      <td>0.684268</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.929719</td>
      <td>0.929719</td>
      <td>0.928607</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.365459</td>
      <td>0.365459</td>
      <td>0.365355</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Observe that boosting with 1 iteration, i.e. the trivial, non-boosting, produces the same posterior probability as the vanilla NB. On the other hand, the first non-trivial boosting differs from them a tiny bit. It has been changed according to the gradient of the vanilla NB.

Now let's apply the boostied NB to some of the publicly available ML datasets. Note that the current version of the boosted NB is __binary classification only__.

### 1.b Framework for comparing the performance

I'll start this subsection by importing the necessary tools and defining functions which will make it easy to compare the performance of several ML classification algorithms, including `NaiveBayesClassifier` and `NaiveBayesBoostingClassifier`. The other ones are:

* `GaussianNB`
* `AdaBoost`
* `GradientBoostingClassifier` from `sklearn`, which is gradient boosted decision trees.

The classifiers listed above are included with their default choice of parameters, without any parameter tuning, as that is beyond the scope of this post. On the other hand, I included two versions of the boosted NB, one with 10 boosting iterations and one with 20. This will help us have a feeling about how the classifier progresses with more iterations.


```python
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer


def make_pipe(est):
    pipe = [('imputer', Imputer()),
            ('estimator', est)]
    return Pipeline(pipe)

def make_boost_pipe(n):
    pipe_boost = [('preprocessor', NaiveBayesPreprocessor(bins=20)),
                  ('nbb', NaiveBayesBoostingClassifier(n_iter=n))]
    pipe_boost = Pipeline(pipe_boost)
    return pipe_boost

%run nb.py
pipe = [('preprocessor', NaiveBayesPreprocessor(bins=20)),
        ('nb', NaiveBayesClassifier())]
pipe = Pipeline(pipe)
algos = {'nbayes': pipe,
        'gnb': make_pipe(GaussianNB()),
        'ada': make_pipe(AdaBoostClassifier()),
        'gbt': make_pipe(GradientBoostingClassifier())}
for i in [10, 20]:
    algos['nbayes+gboost {}'.format(i)] = make_boost_pipe(i)


def compare(X, y):
    rs = ShuffleSplit(test_size=0.25, n_splits=3)
    accuracies = defaultdict(list)
    rocs = defaultdict(list)
    times = {}
    for train_index, test_index in rs.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        for key, est in algos.items():
            then = datetime.now()
            est.fit(X_train, y_train)
            accuracies[key].append(est.score(X_test, y_test))
            # now the roc auc
            if not isinstance(est.steps[-1][-1], GaussianNB):
                ys = est.decision_function(X_test)
                if len(ys.shape) == 2 and ys.shape[1] == 2:
                    ys = ys[1]
            else:
                ys = est.predict_proba(X_test)[:, 1]
            rocs[key].append(roc_auc_score(y_test, ys))
            times[key] = datetime.now() - then
    accuracies = pd.DataFrame(accuracies)
    rocs = pd.DataFrame(rocs)
    times = pd.Series(times)
    accuracies.index.name = 'accuracy'
    rocs.index.name = 'roc-auc'
    times.index.name = 'time'
    return accuracies, rocs, times


def go(X, y):
    accuracies, rocs, times = compare(X, y)
    print(accuracies.to_string())
    print('-'*80)
    print(rocs.to_string())
    print('-'*80)
    print(times.to_string())
```

Quick look at the last line of the `compare` function tells us that it returns the accuracies, roc-auc's and the times it took for all the algorithms it tried. The variable `algos` contains the algorithms we are trying. Now we are in a good position to apply these algorithms to various binary classification problems and see what happens.

---

## 2. Trying it out

In this section, the code cells contain urls which refer to the datasets being used. For more information about them, you can follow those links.

### 2.1 [Spamdata](https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION)

The first choice is the spam/non-spam binary classification for emails. A problem that is well known to be a good use case for classical Naive Bayes approach, although that sort of implies the dataset has *bag-of-words* type features and the NB algorithm in question is either `MultinomialNB` or `BernoulliNB`.

Let's load the dataset, create our `X` and `y`, and take a quick glance at them.


```python
spamdata = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data',
    header=None)
X = spamdata[spamdata.columns[:-1]].rename(columns={i: 'col{}'.format(i) for i in spamdata})
y = spamdata[spamdata.columns[-1]]
print(y.value_counts())
print('-'*80)
print(X.info())
X.sample(3)
```

    0    2788
    1    1813
    Name: 57, dtype: int64
    --------------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4601 entries, 0 to 4600
    Data columns (total 57 columns):
    col0     4601 non-null float64
    col1     4601 non-null float64
    col2     4601 non-null float64
    col3     4601 non-null float64
    col4     4601 non-null float64
    col5     4601 non-null float64
    col6     4601 non-null float64
    col7     4601 non-null float64
    col8     4601 non-null float64
    col9     4601 non-null float64
    col10    4601 non-null float64
    col11    4601 non-null float64
    col12    4601 non-null float64
    col13    4601 non-null float64
    col14    4601 non-null float64
    col15    4601 non-null float64
    col16    4601 non-null float64
    col17    4601 non-null float64
    col18    4601 non-null float64
    col19    4601 non-null float64
    col20    4601 non-null float64
    col21    4601 non-null float64
    col22    4601 non-null float64
    col23    4601 non-null float64
    col24    4601 non-null float64
    col25    4601 non-null float64
    col26    4601 non-null float64
    col27    4601 non-null float64
    col28    4601 non-null float64
    col29    4601 non-null float64
    col30    4601 non-null float64
    col31    4601 non-null float64
    col32    4601 non-null float64
    col33    4601 non-null float64
    col34    4601 non-null float64
    col35    4601 non-null float64
    col36    4601 non-null float64
    col37    4601 non-null float64
    col38    4601 non-null float64
    col39    4601 non-null float64
    col40    4601 non-null float64
    col41    4601 non-null float64
    col42    4601 non-null float64
    col43    4601 non-null float64
    col44    4601 non-null float64
    col45    4601 non-null float64
    col46    4601 non-null float64
    col47    4601 non-null float64
    col48    4601 non-null float64
    col49    4601 non-null float64
    col50    4601 non-null float64
    col51    4601 non-null float64
    col52    4601 non-null float64
    col53    4601 non-null float64
    col54    4601 non-null float64
    col55    4601 non-null int64
    col56    4601 non-null int64
    dtypes: float64(55), int64(2)
    memory usage: 2.0 MB
    None





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
      <th>col0</th>
      <th>col1</th>
      <th>col2</th>
      <th>col3</th>
      <th>col4</th>
      <th>col5</th>
      <th>col6</th>
      <th>col7</th>
      <th>col8</th>
      <th>col9</th>
      <th>...</th>
      <th>col47</th>
      <th>col48</th>
      <th>col49</th>
      <th>col50</th>
      <th>col51</th>
      <th>col52</th>
      <th>col53</th>
      <th>col54</th>
      <th>col55</th>
      <th>col56</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1783</th>
      <td>0.33</td>
      <td>0.84</td>
      <td>0.67</td>
      <td>0.0</td>
      <td>0.67</td>
      <td>0.33</td>
      <td>0.67</td>
      <td>0.0</td>
      <td>0.33</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.183</td>
      <td>0.000</td>
      <td>0.156</td>
      <td>0.104</td>
      <td>0.026</td>
      <td>6.500</td>
      <td>525</td>
      <td>858</td>
    </tr>
    <tr>
      <th>3351</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.751</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.428</td>
      <td>4</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2601</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1.769</td>
      <td>8</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 57 columns</p>
</div>




```python
go(X, y)
```

                   ada       gbt       gnb    nbayes  nbayes+gboost 10  nbayes+gboost 20
    accuracy                                                                            
    0         0.936577  0.946134  0.810599  0.893136          0.919201          0.932233
    1         0.934839  0.947003  0.820156  0.894005          0.933970          0.943527
    2         0.942659  0.947871  0.817550  0.905300          0.931364          0.942659
    --------------------------------------------------------------------------------
                  ada       gbt       gnb    nbayes  nbayes+gboost 10  nbayes+gboost 20
    roc-auc                                                                            
    0        0.975604  0.983871  0.939048  0.565013          0.975606          0.978455
    1        0.977105  0.987682  0.942894  0.545387          0.980848          0.984901
    2        0.979726  0.988510  0.956382  0.527922          0.981500          0.984293
    --------------------------------------------------------------------------------
    time
    ada                00:00:00.250069
    gbt                00:00:00.527525
    gnb                00:00:00.006865
    nbayes             00:00:04.577352
    nbayes+gboost 10   00:00:28.502304
    nbayes+gboost 20   00:00:59.384143


**Evaluation**:

* `GaussianNB` produced around 94% roc auc, which is not a bad number. 
* Vanilla NB produced a 56% roc auc. That somehow translated to a higher accuracy, which suggests a threshold problem with the `GaussianNB`.
* Classical boosting algorithms got to 98% roc auc.
* Boosted NB with 10 iterations improved significantly over Vanilla NB, and it achieved 98% roc auc. 20 iterations improved slighlty over 10 iterations, which is not a bad sign.

So in this case, boosted NB did very comparatively with AdaBoost and Gradient Boosted DTs. Moreover, we observed a good improvement provided by the boosting iterations over the vanilla NB algorithm. We hope this to be a recurring theme.

---

### 2.2 [Bankruptcy Data](http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)

Next up I would like to switch to a more business oriented problem. Here we have the data of Polish companies and the ones that went bankrupt. Let's again load the data and take a quick look. Afterwards, let's apply the algorithms and compare.


```python
# Polish bankruptcy data from
# http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data

from scipy.io import arff


def load_arff(fn):
    X = arff.loadarff(fn)[0]
    X = pd.DataFrame(X).applymap(lambda r: r if r != '?' else None)
    # X.dropna(how='any', inplace=True)
    y = X.pop('class')
    y = (y == b'1').astype(int)
    return X, y


fns = !ls /home/taylanbil/Downloads/*year.arff
_ = [load_arff(fn) for fn in fns]
X = pd.concat([X for X, y in _]).reset_index(drop=True)
y = pd.concat([y for X, y in _]).reset_index(drop=True)
del _
print(y.value_counts())
print('-'*80)
print(X.info())
X.sample(3)
```

    0    41314
    1     2091
    Name: class, dtype: int64
    --------------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 43405 entries, 0 to 43404
    Data columns (total 64 columns):
    Attr1     43397 non-null float64
    Attr2     43397 non-null float64
    Attr3     43397 non-null float64
    Attr4     43271 non-null float64
    Attr5     43316 non-null float64
    Attr6     43397 non-null float64
    Attr7     43397 non-null float64
    Attr8     43311 non-null float64
    Attr9     43396 non-null float64
    Attr10    43397 non-null float64
    Attr11    43361 non-null float64
    Attr12    43271 non-null float64
    Attr13    43278 non-null float64
    Attr14    43397 non-null float64
    Attr15    43369 non-null float64
    Attr16    43310 non-null float64
    Attr17    43311 non-null float64
    Attr18    43397 non-null float64
    Attr19    43277 non-null float64
    Attr20    43278 non-null float64
    Attr21    37551 non-null float64
    Attr22    43397 non-null float64
    Attr23    43278 non-null float64
    Attr24    42483 non-null float64
    Attr25    43397 non-null float64
    Attr26    43310 non-null float64
    Attr27    40641 non-null float64
    Attr28    42593 non-null float64
    Attr29    43397 non-null float64
    Attr30    43278 non-null float64
    Attr31    43278 non-null float64
    Attr32    43037 non-null float64
    Attr33    43271 non-null float64
    Attr34    43311 non-null float64
    Attr35    43397 non-null float64
    Attr36    43397 non-null float64
    Attr37    24421 non-null float64
    Attr38    43397 non-null float64
    Attr39    43278 non-null float64
    Attr40    43271 non-null float64
    Attr41    42651 non-null float64
    Attr42    43278 non-null float64
    Attr43    43278 non-null float64
    Attr44    43278 non-null float64
    Attr45    41258 non-null float64
    Attr46    43270 non-null float64
    Attr47    43108 non-null float64
    Attr48    43396 non-null float64
    Attr49    43278 non-null float64
    Attr50    43311 non-null float64
    Attr51    43397 non-null float64
    Attr52    43104 non-null float64
    Attr53    42593 non-null float64
    Attr54    42593 non-null float64
    Attr55    43404 non-null float64
    Attr56    43278 non-null float64
    Attr57    43398 non-null float64
    Attr58    43321 non-null float64
    Attr59    43398 non-null float64
    Attr60    41253 non-null float64
    Attr61    43303 non-null float64
    Attr62    43278 non-null float64
    Attr63    43271 non-null float64
    Attr64    42593 non-null float64
    dtypes: float64(64)
    memory usage: 21.2 MB
    None





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
      <th>Attr1</th>
      <th>Attr2</th>
      <th>Attr3</th>
      <th>Attr4</th>
      <th>Attr5</th>
      <th>Attr6</th>
      <th>Attr7</th>
      <th>Attr8</th>
      <th>Attr9</th>
      <th>Attr10</th>
      <th>...</th>
      <th>Attr55</th>
      <th>Attr56</th>
      <th>Attr57</th>
      <th>Attr58</th>
      <th>Attr59</th>
      <th>Attr60</th>
      <th>Attr61</th>
      <th>Attr62</th>
      <th>Attr63</th>
      <th>Attr64</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28436</th>
      <td>0.027486</td>
      <td>0.28592</td>
      <td>-0.070966</td>
      <td>0.64665</td>
      <td>-68.132</td>
      <td>0.048296</td>
      <td>0.033511</td>
      <td>2.22740</td>
      <td>1.06920</td>
      <td>0.63686</td>
      <td>...</td>
      <td>-3472.5</td>
      <td>0.064732</td>
      <td>0.043158</td>
      <td>0.93527</td>
      <td>0.1336</td>
      <td>15.0500</td>
      <td>14.057</td>
      <td>90.126</td>
      <td>4.0499</td>
      <td>0.93478</td>
    </tr>
    <tr>
      <th>31905</th>
      <td>0.148640</td>
      <td>0.18727</td>
      <td>0.594540</td>
      <td>10.38300</td>
      <td>353.680</td>
      <td>0.000000</td>
      <td>0.183680</td>
      <td>4.33980</td>
      <td>0.74969</td>
      <td>0.81273</td>
      <td>...</td>
      <td>3424.4</td>
      <td>0.232490</td>
      <td>0.182890</td>
      <td>0.75900</td>
      <td>0.0000</td>
      <td>19.1100</td>
      <td>34.970</td>
      <td>30.850</td>
      <td>11.8320</td>
      <td>2.19150</td>
    </tr>
    <tr>
      <th>11506</th>
      <td>0.001574</td>
      <td>0.50511</td>
      <td>0.084972</td>
      <td>1.17170</td>
      <td>-32.080</td>
      <td>0.000000</td>
      <td>0.002360</td>
      <td>0.97975</td>
      <td>2.94260</td>
      <td>0.49489</td>
      <td>...</td>
      <td>108.0</td>
      <td>-0.004278</td>
      <td>0.003180</td>
      <td>0.99921</td>
      <td>0.0000</td>
      <td>8.6374</td>
      <td>12.808</td>
      <td>61.386</td>
      <td>5.9459</td>
      <td>7.00370</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 64 columns</p>
</div>




```python
go(X, y)
```

                   ada       gbt       gnb    nbayes  nbayes+gboost 10  nbayes+gboost 20
    accuracy                                                                            
    0         0.954386  0.970052  0.078787  0.727055          0.906008          0.932547
    1         0.957704  0.971250  0.065334  0.730280          0.911076          0.943421
    2         0.956782  0.971618  0.067822  0.731755          0.902046          0.946461
    --------------------------------------------------------------------------------
                  ada       gbt       gnb    nbayes  nbayes+gboost 10  nbayes+gboost 20
    roc-auc                                                                            
    0        0.879490  0.912178  0.497303  0.733217          0.760798          0.794332
    1        0.890159  0.921132  0.503383  0.776884          0.786405          0.830294
    2        0.888399  0.925583  0.495057  0.758146          0.777545          0.813152
    --------------------------------------------------------------------------------
    time
    ada                00:00:12.529434
    gbt                00:00:15.847193
    gnb                00:00:00.074148
    nbayes             00:00:52.060308
    nbayes+gboost 10   00:04:27.274454
    nbayes+gboost 20   00:08:38.005423


**Evaluation**:

* Once again, `GaussianNB` performed the worst among the bunch. Not only that, but this time it provided no predictivity, with 50% roc auc.
* Classical boosting algorithms did well, with `AdaBoost` around 89% and `GradientBoostingClassifier` around 92% roc auc.
* Vanilla NB had around 75% roc auc.
* Boosted NB with 10 iterations -> 78% roc auc. Not a ground breaking improvement but an improvement nonetheless.
* 20 iterations got us to 81% roc. This suggests a consistent increase in performance.

In this example, we have another case of boosting iterations improving the performance. As known from classical boosting algorithms, number of iterations is a hyperparameter one needs to tune for a given problem. It does not seem that we have hit a ceiling of performance here. However, tuning those parameters is not the goal of this post, so we'll skip that discussion.

### 2.3 [Credit Card Defaults Data](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

Continuing with the finance theme, third dataset is the credit card default dataset from consumers in Taiwan. Let's dive in.


```python
'https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients'
bankruptcy = pd.read_excel('/home/taylanbil/Downloads/default of credit card clients.xls', skiprows=1)
y = bankruptcy.pop('default payment next month')
X = bankruptcy
print(y.value_counts())
print('-'*80)
print(X.info())
X.sample(3)
```

    0    23364
    1     6636
    Name: default payment next month, dtype: int64
    --------------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 24 columns):
    ID           30000 non-null int64
    LIMIT_BAL    30000 non-null int64
    SEX          30000 non-null int64
    EDUCATION    30000 non-null int64
    MARRIAGE     30000 non-null int64
    AGE          30000 non-null int64
    PAY_0        30000 non-null int64
    PAY_2        30000 non-null int64
    PAY_3        30000 non-null int64
    PAY_4        30000 non-null int64
    PAY_5        30000 non-null int64
    PAY_6        30000 non-null int64
    BILL_AMT1    30000 non-null int64
    BILL_AMT2    30000 non-null int64
    BILL_AMT3    30000 non-null int64
    BILL_AMT4    30000 non-null int64
    BILL_AMT5    30000 non-null int64
    BILL_AMT6    30000 non-null int64
    PAY_AMT1     30000 non-null int64
    PAY_AMT2     30000 non-null int64
    PAY_AMT3     30000 non-null int64
    PAY_AMT4     30000 non-null int64
    PAY_AMT5     30000 non-null int64
    PAY_AMT6     30000 non-null int64
    dtypes: int64(24)
    memory usage: 5.5 MB
    None





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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22959</th>
      <td>22960</td>
      <td>210000</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>89173</td>
      <td>91489</td>
      <td>93463</td>
      <td>95021</td>
      <td>4600</td>
      <td>3789</td>
      <td>3811</td>
      <td>3465</td>
      <td>3186</td>
      <td>4389</td>
    </tr>
    <tr>
      <th>24953</th>
      <td>24954</td>
      <td>60000</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6733</td>
      <td>7662</td>
      <td>8529</td>
      <td>9884</td>
      <td>1500</td>
      <td>1300</td>
      <td>1200</td>
      <td>1000</td>
      <td>1500</td>
      <td>800</td>
    </tr>
    <tr>
      <th>10352</th>
      <td>10353</td>
      <td>360000</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>58</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>1090</td>
      <td>780</td>
      <td>390</td>
      <td>388</td>
      <td>554</td>
      <td>1096</td>
      <td>780</td>
      <td>0</td>
      <td>388</td>
      <td>887</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
go(X, y)
```

                   ada       gbt       gnb    nbayes  nbayes+gboost 10  nbayes+gboost 20
    accuracy                                                                            
    0         0.813600  0.815467  0.410800  0.757333          0.788133          0.791867
    1         0.818933  0.823200  0.365067  0.758667          0.790133          0.795333
    2         0.814667  0.821867  0.371867  0.760933          0.793467          0.799333
    --------------------------------------------------------------------------------
                  ada       gbt       gnb    nbayes  nbayes+gboost 10  nbayes+gboost 20
    roc-auc                                                                            
    0        0.762167  0.770514  0.658582  0.483075          0.745817          0.739989
    1        0.779408  0.783143  0.677352  0.479125          0.762096          0.754629
    2        0.779661  0.787814  0.659249  0.483573          0.758837          0.750321
    --------------------------------------------------------------------------------
    time
    ada                00:00:01.701464
    gbt                00:00:02.663667
    gnb                00:00:00.023219
    nbayes             00:00:13.379899
    nbayes+gboost 10   00:02:21.470685
    nbayes+gboost 20   00:04:33.042267


**Evaluation**:

* Similar as above, `AdaBoost` and `GBT` are the best ones.
* `GaussianNB` did ok.
* Vanilla NB had atrocious performance. 48% roc auc.
* Boosted NB with 10 iterations had 76% roc auc. Pretty good performance.
* 20 iterations had slightly worse performance; 75% roc auc.

This case illustrates the importance of boosting very well. With 1 iteration, NB has absolutely no predictive value. However, once we start boosting, we get to a pretty good performance, competing with the classical boosting algorithms! That is a striking difference. 20 iterations being worse than 10 suggests that the optimal value is less than 20 (it could be less than 10 too, we did not check that). 

---

All in all, we saw the improvement offered by boosting the basic NB algorithm. Although it did not beat the boosted DTs in these examples, there may be cases where it does. There seems to be value in adding this algorithm to one's ML toolkit.

Now let's get down to the nitty-gritty and see how this all works.

## 3. Code

The code can be found [here](https://github.com/taylanbil/naivebayes/blob/master/nb.py). I will paste bits and pieces from that file and try to explain how it comes together.

### 3.1. Review of vanilla NB

First of all, the boosted NB builds on the classes introduced in the [previous post](/vanillanb). For convenience, here's the code from that discussion.

As a quick reminder; the code contains three major parts;

1. `NaiveBayesClassifier`, which implements the NB algorithm.
2. `NaiveBayesPreprocessor`, which fits and/or transforms a dataset so that the output is suitable for applying the `NaiveBayesClassifier`.
3. `Discretizer`, which takes a continuous `pandas` Series and bins it. It also remembers the bins for future use (typically @ scoring time). This is employed by the `NaiveBayesPreprocessor`.


```python
from collections import defaultdict
from operator import itemgetter
from bisect import bisect_right

import numpy as np
import pandas as pd


class Discretizer(object):

    def __init__(self, upperlim=20, bottomlim=0, mapping=False):
        self.mapping = mapping
        self.set_lims(upperlim, bottomlim)

    @property
    def cutoffs(self):
        return [i[0] for i in self.mapping]

    def set_lims(self, upperlim, bottomlim):
        if not self.mapping:
            self.bottomlim = bottomlim
            self.upperlim = upperlim
        else:
            vals = sorted(np.unique(map(itemgetter(1), self.mapping)))
            self.bottomlim = vals[0]
            self.upperlim = vals[-1]
        assert self.bottomlim < self.upperlim

    def fit(self, continuous_series, subsample=None):
        self.mapping = []
        continuous_series = pd.Series(continuous_series).reset_index(drop=True).dropna()
        if subsample is not None:
            n = len(continuous_series)*subsample if subsample < 1 else subsample
            continuous_series = np.random.choice(continuous_series, n, replace=False)
        ranked = pd.Series(continuous_series).rank(pct=1, method='average')
        ranked *= self.upperlim - self.bottomlim
        ranked += self.bottomlim
        ranked = ranked.map(round)
        nvals = sorted(np.unique(ranked))  # sorted in case numpy changes
        for nval in nvals:
            cond = ranked == nval
            self.mapping.append((continuous_series[cond].min(), int(nval)))

    def transform_single(self, val):
        if not self.mapping:
            raise NotImplementedError('Haven\'t been fitted yet')
        elif pd.isnull(val):
            return None
        i = bisect_right(self.cutoffs, val) - 1
        if i == -1:
            return 0
        return self.mapping[i][1]

    def transform(self, vals):
        if isinstance(vals, float):
            return self.transform_single(vals)
        elif vals is None:
            return None
        return pd.Series(vals).map(self.transform_single)

    def fit_transform(self, vals):
        self.fit(vals)
        return self.transform(vals)


class NaiveBayesPreprocessor(object):
    """
    Don't pass in Nans. fill with keyword.
    """

    OTHER = '____OTHER____'
    FILLNA = '____NA____'

    def __init__(self, alpha=1.0, min_freq=0.01, bins=20):
        self.alpha = alpha  # Laplace smoothing
        self.min_freq = min_freq  # drop values occuring less frequently than this
        self.bins = bins  # number of bins for continuous fields

    def learn_continuous_transf(self, series):
        D = Discretizer(upperlim=self.bins)
        D.fit(series)
        return D

    def learn_discrete_transf(self, series):
        vcs = series.value_counts(dropna=False, normalize=True)
        vcs = vcs[vcs >= self.min_freq]
        keep = set(vcs.index)
        transf = lambda r: r if r in keep else self.OTHER
        return transf

    def learn_transf(self, series):
        if series.dtype == np.float64:
            return self.learn_continuous_transf(series)
        else:
            return self.learn_discrete_transf(series)

    def fit(self, X_orig, y=None):
        """
        Expects pandas series and pandas DataFrame
        """
        # get dtypes
        self.dtypes = defaultdict(set)
        for fld, dtype in X_orig.dtypes.iteritems():
            self.dtypes[dtype].add(fld)

        X = X_orig
        # X = X_orig.fillna(self.FILLNA)
        # get transfs
        self.transformations = {
            fld: self.learn_transf(series)
            for fld, series in X.iteritems()}

    def transform(self, X_orig, y=None):
        """
        Expects pandas series and pandas DataFrame
        """
        X = X_orig.copy()
        # X = X_orig.fillna(self.FILLNA)
        for fld, func in self.transformations.items():
            if isinstance(func, Discretizer):
                X[fld] = func.transform(X[fld])
            else:
                X[fld] = X[fld].map(func)
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class NaiveBayesClassifier(object):

    def __init__(self, alpha=1.0, class_priors=None, **kwargs):
        self.alpha = alpha
        self.class_priors = class_priors

    def get_class_log_priors(self, y):
        self.classes_ = y.unique()
        if self.class_priors is None:
            self.class_priors = y.value_counts(normalize=1)
        elif isinstance(self.class_priors, str) and self.class_priors == 'equal':
            raise NotImplementedError
        self.class_log_priors = self.class_priors.map(np.log)

    def get_log_likelihoods(self, fld):
        table = self.groups[fld].value_counts(dropna=False).unstack(fill_value=0)
        table += self.alpha
        sums = table.sum(axis=1)
        likelihoods = table.apply(lambda r: r/sums, axis=0)
        log_likelihoods = likelihoods.applymap(np.log)
        return {k if pd.notnull(k) else None: v for k, v in log_likelihoods.items()}

    def fit(self, X, y):
        y = pd.Series(y)
        self.get_class_log_priors(y)
        self.groups = X.groupby(y)
        self.log_likelihoods = {
            fld: self.get_log_likelihoods(fld)
            for fld, series in X.iteritems()
        }

    def get_approx_log_posterior(self, series, class_):
        log_posterior = self.class_log_priors[class_]  # prior
        for fld, val in series.iteritems():
            # there are cases where the `val` is not seen before
            # as in having a `nan` in the scoring dataset,
            #   but no `nans in the training set
            # in those cases, we want to not add anything to the log_posterior

            # This is to handle the Nones and np.nans etc.
            val = val if pd.notnull(val) else None

            if val not in self.log_likelihoods[fld]:
                continue
            log_posterior += self.log_likelihoods[fld][val][class_]
        return log_posterior

    def decision_function_series(self, series):
        approx_log_posteriors = [
            self.get_approx_log_posterior(series, class_)
            for class_ in self.classes_]
        return pd.Series(approx_log_posteriors, index=self.classes_)

    def decision_function_df(self, df):
        return df.apply(self.decision_function_series, axis=1)

    def decision_function(self, X):
        """
        returns the log posteriors
        """
        if isinstance(X, pd.DataFrame):
            return self.decision_function_df(X)
        elif isinstance(X, pd.Series):
            return self.decision_function_series(X)
        elif isinstance(X, dict):
            return self.decision_function_series(pd.Series(X))

    def predict_proba(self, X, normalize=True):
        """
        returns the (normalized) posterior probability

        normalization is just division by the evidence. doesn't change the argmax.
        """
        log_post = self.decision_function(X)
        if isinstance(log_post, pd.Series):
            post = log_post.map(np.exp)
        elif isinstance(log_post, pd.DataFrame):
            post = log_post.applymap(np.exp)
        else:
            raise NotImplementedError('type of X is "{}"'.format(type(X)))
        if normalize:
            if isinstance(post, pd.Series):
                post /= post.sum()
            elif isinstance(post, pd.DataFrame):
                post = post.div(post.sum(axis=1), axis=0)
        return post

    def predict(self, X):
        probas = self.decision_function(X)
        if isinstance(probas, pd.Series):
            return np.argmax(probas)
        return probas.apply(np.argmax, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(np.array(y) == preds.values)
```

### 3.2. Quick Review of boosting

The idea of boosting consists of these following major parts;

1. A weak (or "base") learner which fits to the dataset.
2. A loss function which evaluates the predictions and the truth values.
3. A new weak learner that is trained on the new "truth values", which are adjusted so that the past mistakes are weighed more heavily. The idea is to try to correct the mistakes iteratively.
4. A method (usually line search) that will define how to bring the new weak learner that is freshly trained into the mix of the past weak learners.

To implement this with the NB classifier as the weak learner, we first need a new version of our NB classifier, that can be trained with arbitrary values as the truth set. But the NB algotihm is naturally a classifier. The classical boosting algorithms as `AdaBoost` and `GBT`'s use "regression trees", which are decision trees that output continuous values, as their weak learners. Modifying NB to do that seems a bit awkward. Therefore, I chose to not touch the natural "classifier" aspect of NB. Instead, I coded a version of the NB classifier with sample weights. That is, the samples that are gotten wrong by the past weak learners are weighed heavily in the next iteration.  

Our loss function will be the same as the `BinomialDeviance` loss function that `GBT`s use in `sklearn`.

### 3.3. Code for the Boosted NB

First let's introduce the weak learner, i.e. the `WeightedNaiveBayesClassifier`. This inherits from the `NaiveBayesClassifier`, and modifies the key methods to accomodate sample weights. The code differs in the way that it stores the log likelihoods and the class priors. The scoring / predicting methods are the same as they retrieve data from the log likelihoods and class priors only.


```python
class WeightedNaiveBayesClassifier(NaiveBayesClassifier):

    def get_class_log_priors(self, y, weights):
        self.classes_ = y.unique()
        self.class_priors = {
            class_: ((y == class_)*weights).mean()
            for class_ in self.classes_}
        self.class_log_priors = {
            class_: np.log(prior)
            for class_, prior in self.class_priors.items()}

    def get_log_likelihood(self, y, class_, series, val, vals, weights):
        # indices of `series` and `y` and `weights` must be the same
        # XXX: what to do with alpha? should we smooth it out somehow?
        cond = y == class_
        num = ((series.loc[cond] == val) * weights.loc[cond]).sum() + self.alpha
        denom = (weights.loc[cond]).sum() + len(vals) * self.alpha
        return np.log(num/denom)

    def get_log_likelihoods(self, series, y, weights):
        vals = series.unique()
        y_ = y.reset_index(drop=True, inplace=False)
        series_ = series.reset_index(drop=True, inplace=False)
        weights_ = weights.reset_index(drop=True, inplace=False)

        log_likelihoods = {
            val: {class_: self.get_log_likelihood(y_, class_, series_, val, vals, weights_)
                  for class_ in self.classes_}
            for val in vals}
        return log_likelihoods

    def fit(self, X, y, weights=None):
        if weights is None:
            return super(WeightedNaiveBayesClassifier, self).fit(X, y)
        weights *= len(weights) / weights.sum()
        y = pd.Series(y)
        self.get_class_log_priors(y, weights)
        self.log_likelihoods = {
            fld: self.get_log_likelihoods(series, y, weights)
            for fld, series in X.iteritems()
        }
```

Now we're ready for the final piece, the `NaiveBayesBoostingClassifier`. Here's the code.


```python
class NaiveBayesBoostingClassifier(object):
    """
    For now, this is Binary classification only.
    `y` needs to consist of 0 or 1
    """

    TRUTHFLD = '__'

    def __init__(self, alpha=1.0, n_iter=5, learning_rate=0.1):
        self.n_iter = n_iter
        self.alpha = alpha
        self.learning_rate = learning_rate

    def loss(self, y, pred, vectorize=False):
        """
        This uses the `BinomialDeviance` loss function.
        That means, this implementation of NaiveBayesBoostingClassifier
            is for binary classification only. Change this function
            to implement multiclass classification

        Compute the deviance (= 2 * negative log-likelihood).

        note that `pred` here is the actual predicted posterior proba.
        `pred` in sklearn is log odds
        """
        logodds = np.log(pred/(1-pred))
        # logaddexp(0, v) == log(1.0 + exp(v))
        ans = -2.0 * ((y * logodds) - np.logaddexp(0.0, logodds))
        return ans if vectorize else np.mean(ans)

    def adjust_weights(self, X, y, weights):
        if weights is None:
            return pd.Series([1]*len(y), index=y.index)
            # return -- this errors out bc defers to non weighted nb above

        pred = self.predicted_posteriors
        return self.loss(y, pred, vectorize=True)

    def line_search_helper(self, new_preds, step):
        if self.predicted_posteriors is None:
            return new_preds
        ans = self.predicted_posteriors * (1-step)
        ans += step * new_preds
        return ans

    def line_search(self, new_preds, y):
        # TODO: This can be done using scipy.optimize.line_search
        # but for now, we'll just try 10 values and pick the best one
        if not self.line_search_results:
            self.line_search_results = [1]
            return 1
        steps_to_try = -1 * np.arange(
            -self.learning_rate, 0, self.learning_rate/10)
        step = min(
            steps_to_try,
            key=lambda s: self.loss(
                y, self.line_search_helper(new_preds, s))
        )
        self.line_search_results.append(step)
        return step

    def _predict_proba_1(self, est, X):
        # XXX: another place where we assume binary clf
        # TODO: need a robust way to get 1
        return est.predict_proba(X)[1]

    def fit(self, X, y, weights=None):
        self.stages = []
        self.predicted_posteriors = None
        self.line_search_results = []
        weights = None
        for i in range(self.n_iter):
            weights = self.adjust_weights(X, y, weights)
            nbc = WeightedNaiveBayesClassifier(alpha=self.alpha)
            nbc.fit(X, y, weights)
            new_preds = self._predict_proba_1(nbc, X)
            self.stages.append(nbc)
            self.line_search(new_preds, y)
            self.predicted_posteriors = self.line_search_helper(
                new_preds, self.line_search_results[-1])

    def decision_function_df(self, X, staged=False):
        stage_posteriors = [
            self._predict_proba_1(est, X) for est in self.stages]
        posteriors = 0
        if staged:
            staged_posteriors = dict()
        for i, (sp, step) in enumerate(zip(stage_posteriors, self.line_search_results)):
            posteriors = (1-step)*posteriors + step*sp
            if staged:
                staged_posteriors[i] = posteriors
        return posteriors if not staged else pd.DataFrame(staged_posteriors)

    def decision_function(self, X):
        if isinstance(X, pd.DataFrame):
            return self.decision_function_df(X)
        else:
            raise NotImplementedError

    def predict(self, X, thr=0.5):
        probas = self.decision_function(X)
        return (probas >= thr).astype(int)
        # # draft for multiclass approach below
        # if isinstance(probas, pd.Series):
        #     return np.argmax(probas)
        # return probas.apply(np.argmax, axis=1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(np.array(y) == preds.values)
```

Perhaps the most interesting part of this code is the `fit` method. Here we see the for loop which builds the NB classifiers for each boosting iteration. The `adjust_weights` method computes the new sample weights according to the loss function, which then is fed to the `WeightedNaiveBayesClassifier` for the next weak learner.

The way the loss function and the `adjust_weights` method are coded make them restricted to the binary classification case. One would need to replace binomial deviance loss with multinomial deviance loss and modify the weights adjustment accordingly to accomodate the multiclass case.

Then comes the line search part. Although it is possible to do an actual line search here, I opted out for a simple "try 10 things and pick the best one" method. Not the best approach for sure.

Feel free to contact me (@taylanbil in twitter, linkedin, facebook, etc) for any questions/comments.


```python

```
