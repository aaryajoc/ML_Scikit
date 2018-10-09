from sklearn.datasets import make_moons
import pandas as pd
from sklearn import svm

X,y=make_moons(n_samples=500) #get dataset 500 samples
df=pd.DataFrame(data={'X1' : X[:,0], 'X2' : X[:,1], 'y' : y}) #make it like excel sheet
df=df.sample(frac=1.0) #get sample
train=df.sample(frac=0.5) #get 50% data for training
val=df[~df.index.isin(train.index)] #rest 50% as validation
test=val.sample(frac=0.7) #testing data as 70% of validation data
val=val[~val.index.isin(test.index)] #30% in valdation itself
test[‘y_predicted’]=None #result is empty before we run

#hint: you can use df[‘col’].tolist() and df[‘col’].values to get list and np #array

# your code goes here


train.to_csv('moons_train.csv', index=False, index_label=False)
test.to_csv('moons_test.csv', index=False, index_label=False)
df.to_csv('moons.csv', index=False, index_label=False)
