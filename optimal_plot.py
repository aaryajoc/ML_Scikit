from sklearn.datasets import make_moons, make_blobs
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Getting dataset and sampling for training and validation
X,y=make_moons(n_samples=500)
# X,y=make_blobs(n_samples=500)
df=pd.DataFrame(data={'X1' : X[:,0], 'X2' : X[:,1], 'y' : y})
df=df.sample(frac=1.0)
train= df.sample(frac=0.5)
val=df[~df.index.isin(train.index)]
test=val.sample(frac=0.7)
val=val[~val.index.isin(test.index)]
test['y_predicted_linear']=None 
test['y_predicted_rbf']=None 
X_columns = ["X1","X2"]

# list to iterate through hyperparameters
cost_value = list(range(1,11))
result_matrix = np.zeros((10,10), dtype='float')
kernel = ["linear"]
value = []
# Iterate to find all accuracy values for all hyperparameters
for ker in kernel:
	for each in cost_value:
		for every in cost_value:
			svc_model = SVC(kernel=ker, C=each/10, gamma=every/10)
			svc_model.fit(train[X_columns],train['y'])
			y_new = svc_model.predict(val[X_columns])
			if(accuracy_score(val['y'],y_new) == 1):
				break
			result_matrix[each-1][every-1] = accuracy_score(val['y'], y_new)

	# Find maximum accuracy values to get optimal gamma and C
	max_tuples = np.where(result_matrix == max(map(max, result_matrix)))
	optimal_c = (max_tuples[0][0]+1)/10
	optimal_gamma = (max_tuples[1][0]+1)/10
	svc_model_test = SVC(kernel=ker , C=optimal_c, gamma=optimal_gamma)
	svc_model_test.fit(train[X_columns],train['y'])
	w = svc_model_test.coef_[0]
	a = -w[0] / w[1]
	xx = np.linspace(-5, 5)
	yy = a * xx - (svc_model_test.intercept_[0]) / w[1]

	# value[ker] = svc_model.predict(test[X_columns])
	y_predicted = svc_model.predict(test[X_columns])


plt.figure(figsize=(15,7))
sns.scatterplot(x="X1",y="X2",hue=y_predicted,data=test)
plt.show()


# for each in xx:
# 	print(each)
# print()
# print()
# for each in yy:
	# print(each)

# test['y_predicted_linear'] = value['linear']
# test['y_predicted_rbf'] = value['rbf']
# test.to_csv('moons_test.csv', index=False, index_label=False)
