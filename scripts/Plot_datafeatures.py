import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = "/afs/inf.ed.ac.uk/user/s17/s1749267/mlpproject/data/"
DATA_FILE = "creditcard.csv"

data = pd.read_csv(DATA_DIR+DATA_FILE)
stats = pd.DataFrame({'mean':data.mean(), 'std':data.std(), 'skew':data.skew()})

# normalize the data
data['Amount'] = data['Amount'].apply(lambda x: np.log(x+1e-6))
data.loc[:,'V1':'V28'] = data.loc[:,'V1':'V28'].apply(lambda x: (x-x.mean())/x.std())
#data.loc[:,'V1':'V28'] = data.loc[:,'V1':'V28'].apply(lambda x: (x-x.min())/(x.std(x.max()-x.min())))

def plot(data, key, title):
    pos_data = data[data["Class"]==1]
    neg_data = data[data["Class"]==0]
    fig = plt.figure(figsize=(12,2))
    ax = fig.add_subplot(111)
    ax.hist(pos_data[key], histtype="step", density=True, bins=50, label="pos")
    ax.hist(neg_data[key], histtype="step", density=True, bins=50, label="neg")
    if key == 'Time':
        ax.set_xlim(0,175000)
    elif key == 'Amount':
        ax.set_xlim(-6, 10)
    else:
        ax.set_xlim(-3,3)
    ax.set_title(title)
    ax.set_ylabel("Probability Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(key+".pdf")
    return fig, ax


pos_data = data[data["Class"]==1]
neg_data = data[data["Class"]==0]
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.scatter(neg_data['Time'], neg_data['Amount'], label='neg', alpha=0.2, color='#ff770e')
ax.scatter(pos_data['Time'], pos_data['Amount'], label='pos', alpha=0.2, color='#1f77b4')
ax.set_xlabel('Time')
ax.set_ylabel('log Amount')
ax.set_xlim(0, 175000)
ax.set_ylim(-5, 10)
ax.legend()
fig.tight_layout()
fig.savefig("TimeVsAmount.png")


plot(data, 'Time', 'Time')
plot(data, 'Amount', 'log Amount')
for key in data.keys()[1:-1]:
    plot(data, key, key)
