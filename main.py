import pandas as pd
import warnings
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn import svm

# ignorowanie warningow zwiazanych z roznymi wersjami bibliotek
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# glowna funkcja trenujaca model, obliczajaca czas trenowania i sprawdzajaca jego skutecznosc
def train_predict(clf, x_train, y_train, x_test, y_test):
    print("Trenowanie {}".format(clf.__class__.__name__))
    start = time()
    # trenowanie modelu
    clf.fit(x_train, y_train)
    end = time()
    print("Czas trenowania modelu:  {:.4f} sekund".format(end - start))

    # przewidywanie
    print("Wynik F - score dla:")
    y_pred = clf.predict(x_train)
    print(" - zbioru trenującego: {:.4f}.".format(f1_score(y_train.values, y_pred, pos_label=1)))
    y_pred = clf.predict(x_test)
    print(" - zbioru testującego: {:.4f}.".format(f1_score(y_test.values, y_pred, pos_label=1)))


# wczytanie danych
parkinson_data = pd.read_csv("parkinsons.csv")
n_patients = parkinson_data.shape[0]
n_features = parkinson_data.shape[1] - 1
n_ill = parkinson_data[parkinson_data['status'] == 1].shape[0]
n_healthy = parkinson_data[parkinson_data['status'] == 0].shape[0]

print("\nWczytane dane pacjentow:")
print(" - ilość pacjentów: {}".format(n_patients))
print(" - ilość cech: {}".format(n_features))
print(" - ilość pacjentów chorych: {}".format(n_ill))
print(" - ilość pacjentów zdrowych: {}".format(n_healthy))

feature_cols = list(parkinson_data.columns[1:16]) + list(parkinson_data.columns[18:])
target_col = parkinson_data.columns[17]

x_all = parkinson_data[feature_cols]
y_all = parkinson_data[target_col]

print("\n")
print(x_all)

# standaryzacja danych
x_all = StandardScaler().fit_transform(x_all)

# podział na zbiór trenujący i uczący
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=545)

print("Zbiór trenujący: {} ".format(x_train.shape[0]))
print("Zbiór testujący: {} ".format(x_test.shape[0]))

# wybor klasyfikatorów
clf_gaussianNB = GaussianNB()
clf_SVC = svm.SVC(gamma='auto')
clf_SGD = SGDClassifier(max_iter=1000, loss="hinge")
clf_gradientB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

print("\n-----------------------------------------\n")
print("Naiwny klasyfikator bayesowski:")
train_predict(clf_gaussianNB, x_train, y_train, x_test, y_test)

print("\n-----------------------------------------\n")
print("Maszyna wektorów nośnych:")
train_predict(clf_SVC, x_train, y_train, x_test, y_test)

print("\n-----------------------------------------\n")
print("Zrównoleglona optymalizacja stochastyczna:")
train_predict(clf_SGD, x_train, y_train, x_test, y_test)

print("\n-----------------------------------------\n")
print("Gradient Tree Boosting:")
train_predict(clf_gradientB, x_train, y_train, x_test, y_test)