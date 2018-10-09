from sklearn.datasets import make_moons
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Getting dataset and sampling for training and validation
X,y=make_moons(n_samples=500)
df=pd.DataFrame(data={'X1' : X[:,0], 'X2' : X[:,1], 'y' : y})
df=df.sample(frac=1.0)
train= df.sample(frac=0.5)
val=df[~df.index.isin(train.index)]
test=val.sample(frac=0.7)
val=val[~val.index.isin(test.index)]
test['y_predicted']=None 
X_columns = ["X1","X2"]

# list to iterate through hyperparameters
cost_value = list(range(1,11))
result_matrix = np.zeros((10,10), dtype='float')
kernel = ["linear","rbf"]

# Iterate to find all accuracy values for all hyperparameters
for each in cost_value:
	for every in cost_value:
		svc_model = SVC(kernel="linear", C=each/10, gamma=every/10)
		svc_model.fit(train[X_columns],train['y'])
		y_new = svc_model.predict(val[X_columns])
		if(accuracy_score(val['y'],y_new) == 1):
			break
		print(accuracy_score(val['y'],y_new), ", ", end="", flush=True)
		result_matrix[each-1][every-1] = accuracy_score(val['y'], y_new)
	print()

# Find maximum accuracy values to get optimal gamma and C
max_tuples = np.where(result_matrix == max(map(max, result_matrix)))
optimal_c = (max_tuples[0][0]+1)/10
optimal_gamma = (max_tuples[1][0]+1)/10
svc_model_test = SVC(kernel="linear" , C=optimal_c, gamma=optimal_gamma)
svc_model_test.fit(train[X_columns],train['y'])
y_test = svc_model.predict(test[X_columns])
print(accuracy_score(test['y'],y_test))

# Plot data
fig, axes = plt.subplots()
axes.contour(test[X_columns], test['y'], test[X_columns], colors='k', levels=[-1,0,-1], alpha=0.8, linestyles=['--','-','--'])
# axes.plot(result_matrix[0],list(range(1,11)),'b')
axes.set_title("Moons Test Data for Linear SVM")
fig.savefig("plot.png")

return(results)


#hint: you can use df[‘col’].tolist() and df[‘col’].values to get list and np #array

# your code goes here
# train.to_csv('moons_train.csv', index=False, index_label=False)
test.to_csv('moons_test.csv', index=False, index_label=False)
# df.to_csv('moons.csv', index=False, index_label=False)

# print (train.head())
