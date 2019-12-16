# To use this experimental feature, we need to explicitly ask for it:
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
import sys
import numpy as np
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.datasets import load_iris

np.set_printoptions(threshold=sys.maxsize)
X, y = load_iris(return_X_y=True)
# clf = HistGradientBoostingClassifier(loss='categorical_crossentropy', epsilon_dp_leaves=None).fit(X, y)
# clf = HistGradientBoostingRegressor(verbose=False, max_bins=3).fit(X, y)
clf = HistGradientBoostingClassifier(verbose=False, max_bins=4, epsilon_dp_noise_first=0.5, delta_dp_noise_first=None).fit(X, y)
print(clf.epsilon_dp_noise_first)
print(clf.delta_dp_noise_first)
a = clf.score(X, y)
print(a)
