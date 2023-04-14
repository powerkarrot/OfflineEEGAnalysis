# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])

clf = LDA()
clf.fit(X, y)
LDA()
print(clf.predict([[-0.8, -1]]))

clf2 = LDA()
clf2.coef_ = clf.coef_
LDA()
print(clf2.predict([[-0.8, -1]]))



# %%
