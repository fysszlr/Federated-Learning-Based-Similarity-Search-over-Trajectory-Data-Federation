import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('/home/zlr/t2vec/data/porto.csv')
length = np.zeros([152,])
for subline in tqdm(data['POLYLINE'],desc='Loading Sublines',unit='lines'):
    p = min(len(json.loads(subline)),151)
    length[p] = length[p] + 1
for i,item in enumerate(length):
    print(i,item)
x = ['[0,20]','(20,30]','(30,50]','(50,70]','(70,100]','(100,)']
y = [0,0,0,0,0,0]
for i,item in enumerate(length):
    if 0 <= i and i <= 20 :
        y[0] = y[0] + item
    elif 20 < i and i <= 30 :
        y[1] = y[1] + item
    elif 30 < i and i <= 50 :
        y[2] = y[2] + item
    elif 50 < i and i <= 70 :
        y[3] = y[3] + item
    elif 70 < i and i <= 100 :
        y[4] = y[4] + item
    else:
        y[5] = y[5] + item
plt.bar(x,y)
plt.show()