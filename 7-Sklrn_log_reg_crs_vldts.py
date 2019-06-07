from sklearn.datasets import load_wine

wine = load_wine()
columns_names = wine.feature_names
y = wine.target
X = wine.data

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)

# Cross validation using cross_val_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
print(cross_val_score(lr, X, y, cv=5))

# Cross validation using shuffle split
cv = ShuffleSplit(n_splits=5)
print(cross_val_score(lr, X, y, cv=cv))

