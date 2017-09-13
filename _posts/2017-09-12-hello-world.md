---
title: Hello Worlds
date: 2017-09-12 16:29:00
categories: test
permalink: /hello/
layout: taylandefault
---

hello world  

```python
def bla(x):
    print(x)
    return x**2
```

![myface](/img/nation.png)



```python
import pandas as pd
import numpy as np

X = np.random.randint(0, 10, (100, 3))
X = pd.DataFrame(X)
```


```python
pd.pivot_table(X, index=[0], columns=[2], values=[1], aggfunc=np.sum)
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
    <tr>
      <th></th>
      <th colspan="10" halign="left">1</th>
    </tr>
    <tr>
      <th>2</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.0</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.groupby([0, 2])[1].agg(np.sum).unstack()
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
      <th>2</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
    <tr>
      <th>0</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>9.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>15.0</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.0</td>
      <td>17.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
