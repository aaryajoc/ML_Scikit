import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,7))
sns.scatterplot(x="X1",y="X2",hue="y_predicted",data=test_data_or_some_data)
plt.show()
sns.scatterplot(x="X1",y="X2",hue="y_true",data=test_data_or_some_data)
plt.show()