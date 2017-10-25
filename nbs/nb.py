#!/usr/bin/env python
"""
"""


__author__ = 'taylanbil'


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


class NaiveBayesClassifier2(NaiveBayesClassifier):

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
            return super(NaiveBayesClassifier2, self).fit(X, y)
        weights *= len(weights) / weights.sum()
        y = pd.Series(y)
        self.get_class_log_priors(y, weights)
        self.log_likelihoods = {
            fld: self.get_log_likelihoods(series, y, weights)
            for fld, series in X.iteritems()
        }


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
            nbc = NaiveBayesClassifier2(alpha=self.alpha)
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


if __name__ == '__main__':
    pass
