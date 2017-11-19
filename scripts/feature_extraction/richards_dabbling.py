import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Lasso
# from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import sys
from matplotlib.markers import MarkerStyle

data = np.genfromtxt('2015-dataset.csv', delimiter=',', names=True)
# print data['1890']

# for i in xrange(1891):
    # upper = max(data[str(i)])
    # lower = min(data[str(i)])
    # data[str(i)] = np.array([float(j)/(upper-lower) if not (upper-lower) == 0 else 0 for j in data[str(i)]])

"""
# the histogram of the data
n, bins, patches = plt.hist(data['1890'], 50, facecolor='green', alpha=0.75)
plt.title("Distribution of Article Popularity 2015")
plt.xlabel("Popularity Score = # favorites + 2 * # retweets")
plt.ylabel("Number of Articles")
plt.grid(True)
plt.show()
"""

"""
# most highly correlated feature
plt.scatter(data['1'], data['1890'], alpha=0.2)
plt.title("Most Highly Correlated Feature versus Popularity Score")
plt.xlabel("Number of Multimedia Elements in the Article")
plt.ylabel("Popularity Score")
"""

"""
# second most highly correlated feature
plt.scatter(data['1753'], data['1890'], alpha=0.2)
plt.title("Second Most Highly Correlated Feature versus Popularity Score")
plt.xlabel("Is an Article (as opposed to Blogpost or Multimedia)")
plt.ylabel("Popularity Score")
plt.show()
"""

correlations = [np.correlate(data[str(i)], data['196'])[0] for i in xrange(196)]

sorted_correlations = [i[0] for i in sorted(enumerate(correlations), reverse=True, key=lambda x:x[1])]
sorted_correlation_values = [i[1] for i in sorted(enumerate(correlations), reverse=True, key=lambda x:x[1])]

print sorted_correlations[0]
print sorted_correlations[1]

"""
# correlations of all features
plt.xlabel("Feature Index")
plt.ylabel("Cross-Correlation Score")
plt.title("Correlation of Features with Popularity Score")
plt.bar(np.arange(len(correlations)), correlations, 5) #, width, color='r', yerr=men_std)
plt.show()
"""

training_features = []
training_labels = []
testing_features = []
testing_labels = []
TRAIN_TEST_SPLIT = 0.7

selected_indices = [0]# sorted_correlations #[:50]

for i, row in enumerate(data):
    selected_features = [row[j] for j in selected_indices]
    label = row[-1]
    if i < TRAIN_TEST_SPLIT*len(data):
        training_features.append(selected_features)
        training_labels.append(label)
    else:
        testing_features.append(selected_features)
        testing_labels.append(label)

# pca = PCA(n_components=10)
# pca.fit_transform(training_features)
# pca.transform(testing_features)

clf = SVR(C=5.0, epsilon=0.2, kernel='sigmoid')
# clf = RandomForestRegressor()
# clf = MLPRegressor(max_iter=10000)
clf.fit(training_features, training_labels)



predicted = clf.predict(testing_features)
print clf.score(testing_features, testing_labels)

plt.scatter(testing_labels, predicted)
plt.show()
