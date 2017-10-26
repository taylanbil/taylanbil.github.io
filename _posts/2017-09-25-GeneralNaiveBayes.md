---
title: A General implementation of Naive Bayes Algorithm
date: 2017-09-25 21:18:39
categories: naivebayes
permalink: vanillanb
layout: taylandefault
published: true
---



# sklearn-like api for a general implementation of Naive Bayes

## 0. Intro:

Although the computations that go into it are very tedious Naive Bayes is one of the more accessible ML classification algorithms out there. The reason for that is it is easy to understand, and it makes __sense__, intuitively speaking.

However, the `sklearn` implementations of Naive Bayes are built (excluding `GaussianNB`) with certain assumptions, which make them tough to use without a lot of pre-processing, as we explored in [this post](/multinbvsbinomnb/). This time, I'd like to implement the Naive Bayes algorithm idea for a general input dataset. I will not optimize the code, so it won't be naturally scalable or anything, the goal is just to hack something together quickly.

---

## 1. Refresher on Naive Bayes

Let's start with revisiting the algorithm. In the following dataset, there are 4 columns to predict whether someone will play tennis on that day. Let's take a quick look at this small dataset:


```python
from __future__ import division
import pandas as pd

# A really simple dataset to demonstrate the algorithm
data = pd.read_csv(
    'https://raw.githubusercontent.com/petehunt/c4.5-compiler/master/example/tennis.csv',
    usecols=['outlook', 'temp', 'humidity', 'wind', 'play']
)

data
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>outlook</th>
      <th>temp</th>
      <th>humidity</th>
      <th>wind</th>
      <th>play</th>
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



Given this dataset, we can use Bayesian thinking to predict whether or not someone will play tennis on a new day, given the weather. Specifically, Naive Bayes proceeds as follows:


Let's say, on day **14**, the weather is 

| outlook  | temp  | humidity  | wind  |
|---|---|---|---|
|  Overcast | Mild  | Normal  | Weak  |  


We want to compute the following probabilities:

$$ P(yes \,|\,  overcast, mild, normal, weak)$$

$$ P(no \,|\, overcast, mild, normal, weak) $$

Instead of computing these exactly, we compute proxies of these. Since this part is supposed to be a refresher only, I'll only include the computation of the first case. It goes like this:


$$ P(yes \,|\, o, m, n, w) \approx P(o | y)\times P(m|y) \times P(n|y) \times P(w|y) \times P(y) $$

And the terms on the right side of the equation can be derived from looking at the dataset and counting. For example, 

$$ P(mild \,|\, yes)$$

is just the count of total days with __play__ = _yes_ and __temperature__ = _mild_, divided by the days with __play__ = _yes_. Let's compute;


```python
mild = data.temp == 'Mild'
yes = data.play == 'Yes'

print('P(mild | yes) = {}/{}'.format((mild&yes).sum(), yes.sum()))
```

    P(mild | yes) = 4/9


The final term in the formula above is $P(y)$. That's even simpler to compute, since it is just the count of lines with _yes_ divided by the total number of lines.

Let's put the pieces together by computing proxies for both probabilities with code. We get;


```python
yes = data.play == 'Yes'

o = data.outlook == 'Overcast'
m = data.temp == 'Mild'
n = data.humidity == 'Normal'
w = data.wind == 'Weak'

ans = {
    'yes': yes.sum(),
    'no': (~yes).sum()
}
for elt in [o, m, n, w]:
    ans['yes'] *= (yes & elt).sum() / yes.sum()
    ans['no'] *= ((~yes) & elt).sum() / (~yes).sum()

print('P(yes | o, m, n, w) approximately is {}'.format(ans['yes']))
print('P(no | o, m, n, w) approximately is {}'.format(ans['no']))
```

    P(yes | o, m, n, w) approximately is 0.79012345679
    P(no | o, m, n, w) approximately is 0.0


Before we end this section, notice that the algorithm thinks there's 0 chance that the person does not play tennis under these circumstances. Not only that's a strong opinion, but also it results in numerical difficulties in a real world implementation. To remedy this, one uses Laplace smoothing.

---

## 2. Implementation

### 2.a. Preprocessing

You may have noticed that in our toy dataset, every attribute was categorical. There were no continuous numerical, nor ordinal fields. Also, unlike `sklearn`'s NB algorithms, we did not make any assumptions about the incoming training dataset. Columns were (thought to be) pairwise independent.

As a design choice, we will present a version of Naive Bayes that works with such a training set. Every column will be categorical, and we will implement the computations we carried out above. In practice though, not every training data will consist ofcategorical columns only. To that end, let's first code a `Discretizer` object. The `Discretizer` will do the following:

1. Takes in a continuous `pandas` series and bins the values.
2. Remembers the cutoffs so that it can apply the same transformations later to find which bin the new value belongs to.

Feel free to skip over the code for the usage.


```python
from operator import itemgetter
from bisect import bisect_right

import numpy as np


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
```

The api is similar to `sklearn`, with `fit`, `transform` and `fit_transform` methods. Let's see the usage on a simple example. You can check if the values of y are binned correctly by looking at the cutoffs.


```python
D = Discretizer(upperlim=5)

y = np.random.uniform(0, 1, 1000)
D.fit(y)
print('cutoffs:')
print(D.cutoffs[1:])
print('-'*30)
print('values:')
print(y[:4])
print('-'*30)
print('bins:')
print(D.transform(y[:4]))
```

    cutoffs:
    [0.10316905194019832, 0.28798638771620189, 0.48254030703944994, 0.69792871673000578, 0.89006354032939106]
    ------------------------------
    values:
    [ 0.50908875  0.00113411  0.73404021  0.3087637 ]
    ------------------------------
    bins:
    0    3
    1    0
    2    4
    3    2
    dtype: int64


Now comes the part where we apply the `Discretizer` object to the whole dataset. To that end, we will define a `NaiveBayesPreprocessor` object. If a field is discrete (i.e. categorical), it will leave it (mostly) untouched (in reality, it will eliminate the values that does not occur more than 1% of the time). If the field is continuous, it will bin it as above.


```python
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
```

We will see the preprocessor in action. Before that though, now is the time to code up the `NaiveBayesClassifier`.

### 2.b. NaiveBayesClassifier

The following class will implement the vanilla Naive Bayes algorithm as seen above. I will try to stick to a `sklearn`-like api, with methods such as `fit`, `predict`, `predict_proba` etc. Once again, code is not optimal at all, the goal is to get to something working quickly.


```python
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

## 3. Example usage

### 3.1. Wine quality dataset

First example will work with the following [dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality), hosted @ https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/.

We will load it locally, and predict if a given wine is *red* or *white*. The reader would need to do minimal prework to get the dataset hosted in the link above in this format.

Let's take a quick look at the dataset:


```python
winedata = pd.read_csv('~/metis/github/ct_intel_ml_curriculum/data/Wine_Quality_Data.csv')
print(winedata.shape)
print('-'*30)
print(winedata.color.value_counts(normalize=True))
winedata.sample(3)
```

    (6497, 13)
    ------------------------------
    white    0.753886
    red      0.246114
    Name: color, dtype: float64





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2196</th>
      <td>7.0</td>
      <td>0.29</td>
      <td>0.37</td>
      <td>4.9</td>
      <td>0.034</td>
      <td>26.0</td>
      <td>127.0</td>
      <td>0.99280</td>
      <td>3.17</td>
      <td>0.44</td>
      <td>10.8</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>5327</th>
      <td>6.2</td>
      <td>0.20</td>
      <td>0.33</td>
      <td>5.4</td>
      <td>0.028</td>
      <td>21.0</td>
      <td>75.0</td>
      <td>0.99012</td>
      <td>3.36</td>
      <td>0.41</td>
      <td>13.5</td>
      <td>7</td>
      <td>white</td>
    </tr>
    <tr>
      <th>2911</th>
      <td>9.6</td>
      <td>0.25</td>
      <td>0.54</td>
      <td>1.3</td>
      <td>0.040</td>
      <td>16.0</td>
      <td>160.0</td>
      <td>0.99380</td>
      <td>2.94</td>
      <td>0.43</td>
      <td>10.5</td>
      <td>5</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>



Let's see what this dataset becomes once we transform it with the `NaiveBayesPreprocessor`


```python
NaiveBayesPreprocessor().fit_transform(winedata).sample(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5998</th>
      <td>7</td>
      <td>6</td>
      <td>4</td>
      <td>18</td>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>14</td>
      <td>5</td>
      <td>17</td>
      <td>4</td>
      <td>5</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3876</th>
      <td>13</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>6</td>
      <td>6</td>
      <td>19</td>
      <td>3</td>
      <td>12</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>12</td>
      <td>5</td>
      <td>15</td>
      <td>8</td>
      <td>3</td>
      <td>10</td>
      <td>8</td>
      <td>2</td>
      <td>13</td>
      <td>12</td>
      <td>18</td>
      <td>7</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>



Observe that the last column (which is categorical), is left untouched.

--- 

Now let's try the Naive Bayes algorithms on this.


```python
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from collections import defaultdict
from sklearn.pipeline import Pipeline

pipe = [('preprocessor', NaiveBayesPreprocessor(bins=20)),
        ('nb', NaiveBayesClassifier())]
pipe = Pipeline(pipe)
nbs = {'_vanilla_': pipe,
       'Multinomial': MultinomialNB(),
       'Bernoulli': BernoulliNB(),
       'Gaussian': GaussianNB()}

X, y = winedata[winedata.columns[:-1]], winedata[winedata.columns[-1]]

rs = ShuffleSplit(test_size=0.25, n_splits=5)
scores = defaultdict(list)
for train_index, test_index in rs.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    for key, est in nbs.items():
        est.fit(X_train, y_train)
        scores[key].append(est.score(X_test, y_test))

pd.DataFrame(scores)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bernoulli</th>
      <th>Gaussian</th>
      <th>Multinomial</th>
      <th>_vanilla_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.764308</td>
      <td>0.973538</td>
      <td>0.920615</td>
      <td>0.987077</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.754462</td>
      <td>0.969231</td>
      <td>0.913846</td>
      <td>0.983385</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.777231</td>
      <td>0.971077</td>
      <td>0.924308</td>
      <td>0.988308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.777846</td>
      <td>0.966154</td>
      <td>0.915077</td>
      <td>0.990769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.780308</td>
      <td>0.965538</td>
      <td>0.928615</td>
      <td>0.989538</td>
    </tr>
  </tbody>
</table>
</div>



As you see above, Bernoulli and Multinomial Naive Bayes algorithms aren't performing well, simply because this dataset isn't suitable for their use, as explored in [the previous post](/multinbvsbinomnb/). `GaussianNB` performs OK, but is beaten by our implementation of NB. To satisfy the curious among us, let's throw `GradientBoostingClassifier` and `RandomForestClassifier` (without parameter tuning) at this;





```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

nbs = {'GBT': GradientBoostingClassifier(), 'RF': RandomForestClassifier()}
scores = defaultdict(list)
for train_index, test_index in rs.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    for key, est in nbs.items():
        est.fit(X_train, y_train)
        scores[key].append(est.score(X_test, y_test))

pd.DataFrame(scores)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GBT</th>
      <th>RF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.994462</td>
      <td>0.994462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.993846</td>
      <td>0.995692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.995077</td>
      <td>0.993846</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.993231</td>
      <td>0.993231</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.993231</td>
      <td>0.993846</td>
    </tr>
  </tbody>
</table>
</div>



Yeah these powerful algorithms yield better results. Let's move on to our second example.

### 3.2. Human Activity Recognition using Smartphones

Dataset and description can be found @ https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones


```python
activity_data = pd.read_csv('~/metis/github/ct_intel_ml_curriculum/data/Human_Activity_Recognition_Using_Smartphones_Data.csv')
print(activity_data.shape)
print('-'*30)
print(activity_data.Activity.value_counts(normalize=True))
activity_data.sample(3)
```

    (10299, 562)
    ------------------------------
    LAYING                0.188756
    STANDING              0.185067
    SITTING               0.172541
    WALKING               0.167201
    WALKING_UPSTAIRS      0.149917
    WALKING_DOWNSTAIRS    0.136518
    Name: Activity, dtype: float64





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tBodyAcc-mean()-X</th>
      <th>tBodyAcc-mean()-Y</th>
      <th>tBodyAcc-mean()-Z</th>
      <th>tBodyAcc-std()-X</th>
      <th>tBodyAcc-std()-Y</th>
      <th>tBodyAcc-std()-Z</th>
      <th>tBodyAcc-mad()-X</th>
      <th>tBodyAcc-mad()-Y</th>
      <th>tBodyAcc-mad()-Z</th>
      <th>tBodyAcc-max()-X</th>
      <th>...</th>
      <th>fBodyBodyGyroJerkMag-skewness()</th>
      <th>fBodyBodyGyroJerkMag-kurtosis()</th>
      <th>angle(tBodyAccMean,gravity)</th>
      <th>angle(tBodyAccJerkMean),gravityMean)</th>
      <th>angle(tBodyGyroMean,gravityMean)</th>
      <th>angle(tBodyGyroJerkMean,gravityMean)</th>
      <th>angle(X,gravityMean)</th>
      <th>angle(Y,gravityMean)</th>
      <th>angle(Z,gravityMean)</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5882</th>
      <td>0.275918</td>
      <td>-0.018486</td>
      <td>-0.100284</td>
      <td>-0.993155</td>
      <td>-0.958860</td>
      <td>-0.962892</td>
      <td>-0.993935</td>
      <td>-0.953637</td>
      <td>-0.960433</td>
      <td>-0.934424</td>
      <td>...</td>
      <td>-0.605982</td>
      <td>-0.868852</td>
      <td>0.048834</td>
      <td>0.094434</td>
      <td>-0.922991</td>
      <td>0.715411</td>
      <td>-0.833794</td>
      <td>0.212639</td>
      <td>0.007696</td>
      <td>STANDING</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>0.324749</td>
      <td>-0.012294</td>
      <td>-0.053416</td>
      <td>-0.336888</td>
      <td>0.157011</td>
      <td>-0.387988</td>
      <td>-0.409819</td>
      <td>0.152768</td>
      <td>-0.426130</td>
      <td>-0.069544</td>
      <td>...</td>
      <td>-0.252935</td>
      <td>-0.653851</td>
      <td>-0.294646</td>
      <td>-0.449551</td>
      <td>0.601912</td>
      <td>-0.138953</td>
      <td>-0.790079</td>
      <td>0.242435</td>
      <td>0.002180</td>
      <td>WALKING</td>
    </tr>
    <tr>
      <th>9466</th>
      <td>0.287556</td>
      <td>-0.018570</td>
      <td>-0.108500</td>
      <td>-0.984497</td>
      <td>-0.981637</td>
      <td>-0.987341</td>
      <td>-0.985297</td>
      <td>-0.982858</td>
      <td>-0.987223</td>
      <td>-0.927790</td>
      <td>...</td>
      <td>-0.656437</td>
      <td>-0.892907</td>
      <td>0.034444</td>
      <td>0.195809</td>
      <td>0.352398</td>
      <td>-0.039287</td>
      <td>0.411160</td>
      <td>-0.307855</td>
      <td>-0.682102</td>
      <td>LAYING</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 562 columns</p>
</div>



Because this dataset has negative numbers in it, `MultinomialNB` will not work with it. We can add the minimum value to it and make it work, but again, it just doesn't make a lot of sense to do that.


```python
pipe = [('preprocessor', NaiveBayesPreprocessor(bins=20)),
        ('nb', NaiveBayesClassifier())]
pipe = Pipeline(pipe)
nbs = {'_vanilla_': pipe,
       # 'Multinomial': MultinomialNB(),
       'Bernoulli': BernoulliNB(),
       'Gaussian': GaussianNB()}

X, y = activity_data[activity_data.columns[:-1]], activity_data[activity_data.columns[-1]]

rs = ShuffleSplit(test_size=0.25, n_splits=5)
scores = defaultdict(list)
for train_index, test_index in rs.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    for key, est in nbs.items():
        est.fit(X_train, y_train)
        scores[key].append(est.score(X_test, y_test))

pd.DataFrame(scores)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bernoulli</th>
      <th>Gaussian</th>
      <th>_vanilla_</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.850485</td>
      <td>0.725049</td>
      <td>0.793398</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.840388</td>
      <td>0.699417</td>
      <td>0.792621</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.846214</td>
      <td>0.716893</td>
      <td>0.764660</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.852039</td>
      <td>0.776311</td>
      <td>0.801165</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.853592</td>
      <td>0.739806</td>
      <td>0.801942</td>
    </tr>
  </tbody>
</table>
</div>



This time, `BernoulliNB` does well. This is because it binarizes the dataset prior to fitting the Bernoulli Naive Bayes, and the threshold it uses to binarize is 0. Incidentally, this works well with predicting the activity. In the prior example, this had almost no added value since everything was nonnegative.

At this point, one could do some parameter tuning, play with the possible bin value etc. In any case, this dataset is not a great dataset for the Naive Bayes type algorithms, but I wanted to see how this implementation does in such an example.

---

In the next post, I will explore a weighted version of this implementation of Naive Bayes, and use it as a weak learner in a boosting scheme.
