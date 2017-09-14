---
title: Monty Hall
date: 2017-09-13 17:20:00
categories: montecarlo
permalink: /montyhall/
layout: taylandefault
published: true
---

[The Monty Hall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) is a famous brain teaser, which often is confusing. The problem statement is;  


> Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?	

At first it seems like since there are only two doors left, it would not matter to switch your pick or not. However, careful reasoning using probability implies it is better to switch your decision at that point.  

---

In this post, I would like to establish this advantage using a simulation. To that end, here's the pseudocode of 1 game:


```python
def montyhallgame(switch):
    determine where the car is
    guest picks a door
    host eliminates a door which is not picked and which has a goat
    if switch:
        switch choice
    return success/failure

```

Once we have this in working code, we can then make the computer play a million games and look at the probability of success if we switch vs if we don't switch.  

First things first, let's translate this pseudocode above into python:


```python
import random

def montyhallgame(switch):
    # pick a random index and make that a car
    doors = ['goat' for _ in range(3)]
    car = random.choice(range(len(doors)))  
    doors[car] = 'car'
    # Now the guest makes their pick
    guest_pick = random.choice(range(len(doors)))
    # Now the host...
    host_remove_index = random.choice(
        [i for i in range(len(doors)) if i != guest_pick and doors[i] == 'goat']
    )  # this index points to the door that the host decides to keep in
    host_keep_index = [i for i in range(3) if i != guest_pick and i != host_remove_index][0]
    if switch:
        return doors[host_keep_index]
    else:
        return doors[guest_pick]
    
print(montyhallgame(False))
print(montyhallgame(True))
```

    car
    car


Allright, now we have a function which tells us what the result of the game is.  

Let's play this game a million times with and without switching and see how many goats/cars do we get.


```python
N_TRIALS = 10**6
switch_car_count = 0
noswitch_car_count = 0

for _ in range(N_TRIALS):
    switch_car_count += montyhallgame(True) == 'car'
    noswitch_car_count += montyhallgame(False) == 'car'

print('if we switch, out of {} games, we get {} wins'.format(
    N_TRIALS, switch_car_count))
print('if we don\'t switch, out of {} games, we get {} wins'.format(
    N_TRIALS, noswitch_car_count))
```

    if we switch, out of 1000000 games, we get 666999 wins
    if we don't switch, out of 1000000 games, we get 333350 wins


Look at that difference! Switching the decision doubles the guest's win rate! This confirms the theoretical calculation.

---

Let's make the difference even more striking. Let's modify the above code so that there aren't just 3 doors, but 100 doors, with 99 goats and 1 car. And once the guest picks, the host comes in and removes 98 doors, all with goats behind.

We'll write the function a bit more efficiently, encoding car by 1 and goat by 0.


```python
import numpy as np

def montyhallgame2(switch, n=100):
    car = random.randint(1, n)
    guest_pick = random.randint(1, n)
    if guest_pick != car:
        host_keep = car  # in this case, we can't eliminate the car
    else:
        # this is our way of deleting the guest_pick from host's options to remove
        host_keep = random.randint(1, n-1) 
        host_keep = host_keep + (host_keep >= guest_pick)
    result = host_keep if switch else guest_pick
    return result == car
```


```python
N_TRIALS = 10**6
switch_car_count = 0
noswitch_car_count = 0

for _ in range(N_TRIALS):
    switch_car_count += montyhallgame2(True)
    noswitch_car_count += montyhallgame2(False)

print('if we switch, out of {} games, we get {} wins'.format(
    N_TRIALS, switch_car_count))
print('if we don\'t switch, out of {} games, we get {} wins'.format(
    N_TRIALS, noswitch_car_count))
```

    if we switch, out of 1000000 games, we get 990081 wins
    if we don't switch, out of 1000000 games, we get 9973 wins


Now that is quite a big difference! In the end, it amounts to the following probabilities:  

> P(win without switch) = P(guessing right the first time)  

> P(win with switch) = P(guessing wrong the first time)
