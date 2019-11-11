# To use this experimental feature, we need to explicitly ask for it:
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
import sys
import numpy as np
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.datasets import load_iris

np.set_printoptions(threshold=sys.maxsize)
X, y = load_iris(return_X_y=True)
clf = HistGradientBoostingClassifier(loss='categorical_crossentropy', epsilon_dp_leaves=0.9).fit(X, y)
# clf = HistGradientBoostingRegressor(max_bins=255, epsilon_dp_leaves=0.7).fit(X, y)
print(clf.epsilon_dp_leaves)
a = clf.score(X, y)

print(a)
