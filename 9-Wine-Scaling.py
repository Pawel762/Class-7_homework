from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
columns_names = wine.feature_names
y = wine.target
X = wine.data

print('Pre scaling X')
print(X)

scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

print('Post scaling X')
print(scaled_features)

X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.375)

