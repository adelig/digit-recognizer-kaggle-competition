import pandas as pd
import numpy as np
import seaborn as sns
import os
from time import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer


path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

# PREPROCESSING

# Load the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1)

# Free some space
del train

# Visualize the data
sns.countplot(Y_train)
Y_train.value_counts()

# Check for missing values
X_train.isnull().any().describe()
test.isnull().any().describe()

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
# 1. INSTANTIATE
enc = OneHotEncoder()
# 2. FIT
enc.fit(Y_train.to_frame())
# 3. Transform
Y_train = enc.transform(Y_train.to_frame()).toarray()
Y_train.shape

# Split the train and the validation set for the fitting
# Since we have balanced labels, no need to use stratify = True option in train_test_split function
X_train, X_val, Y_train, Y_val = train_test_split(X_train, 
                                                  Y_train, 
                                                  test_size = 0.1, 
                                                  random_state=2)

# TRAINING AND EVALUATING MODELS

# Decode one hot encoding
def get_number(y):
    numbers = pd.DataFrame(y)
    return numbers.idxmax(axis=1).as_matrix()

y_val = get_number(Y_val)

def train_classifier(clf):    
    start = time()
    if clf == clf_xgb or clf == clf_svm or clf == clf_knn:
        y_train = get_number(Y_train)
    else:
        y_train = Y_train
    if clf == clf_svm or clf == clf_knn:
        clf.fit(X_train_pca, y_train)
    else:
        clf.fit(X_train, y_train)
    end = time()
    print "Trained model in {:.4f} seconds".format(end - start)

def predict_labels(clf):
    start = time()
    if clf == clf_svm or clf == clf_knn:
        Y_pred = clf.predict(X_val_pca)
    else:
        Y_pred = clf.predict(X_val)   
    end = time()
    if clf == clf_xgb or clf == clf_svm or clf == clf_knn:
        y_pred = Y_pred
    else:
        y_pred = get_number(Y_pred)
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(y_val, y_pred, average = 'weighted'), sum(y_val == y_pred) / float(len(y_pred))

def train_predict(clf):
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    train_classifier(clf)
    f1, acc = predict_labels(clf)
    print "F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc)

def n_component_analysis(clf, n):
    y_train = get_number(Y_train)
    start = time()
    pca = PCA(n_components=n)
    print("PCA begin with n_components: {}".format(n));
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    if clf == clf_svm:
        print('SVM begin')
        clf = SVC()
    elif clf == clf_knn:
        print('KNN begin')
        clf = KNeighborsClassifier()
    clf.fit(X_train_pca, y_train)
    accuracy = clf.score(X_val_pca, y_val)
    end = time()
    print("accuracy: {}, time lapse:{}".format(accuracy, int(end-start)))
    return accuracy

def X_train_val_pca(clf):
    n_range = np.zeros(10)
    if clf == clf_svm:
        n_range = np.linspace(0.7, 0.85, num=10).round(2)
    elif clf == clf_knn:
        n_range = np.linspace(1, 200, num=10).round().astype(int)
    accuracy = []
    for n in n_range:
        acc = n_component_analysis(clf, n)
        print('')
        accuracy.append(acc)
    
    plt.plot(n_range, np.array(accuracy), 'b-')
    
    if clf == clf_svm:
        pca = PCA(n_components=n_range[np.argmax(accuracy)], svd_solver ='full')
    elif clf == clf_knn:
        pca = PCA(n_components=n_range[np.argmax(accuracy)])
    
    pca.fit(X_train)
    print(str(pca.n_components_) + ' components selected')  
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)
    
    return X_train_pca, X_val_pca

# MAIN
clf_lr = LinearRegression()
clf_svm = SVC()
clf_xgb = xgb.XGBClassifier()
clf_rf = RandomForestClassifier(n_estimators = 100, n_jobs=-1)  # Saturated after 100 estimators
clf_knn = KNeighborsClassifier()

clf_list = [clf_lr, clf_svm, clf_xgb, clf_rf, clf_knn]

for cl in clf_list:
    if cl == clf_svm or cl == clf_knn:
        X_train_pca, X_val_pca = X_train_val_pca(cl)
    train_predict(cl)
    print ''
    
# Tuning the parameters of XGBoost
parameters = { 'learning_rate' : [0.1, 0.3],
               'n_estimators' : [60, 100],
               'max_depth': [3, 4],
               'min_child_weight': [1, 2],
               'gamma':[0, 0.2],
               'subsample' : [0.8, 1],
               'colsample_bytree' : [0.8, 1],
               'scale_pos_weight' : [0.5, 1],
               'reg_alpha':[0, 1e-5]
             }

clf_xgb = xgb.XGBClassifier()

f1_scorer = make_scorer(f1_score, average = 'weighted')

grid_obj = GridSearchCV(clf_xgb,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

grid_obj = grid_obj.fit(X_train,get_number(Y_train))

clf_xgb = grid_obj.best_estimator_
print clf_xgb
    
f1, acc = predict_labels(clf_xgb)
print "F1 score and accuracy score for test set after tuning: {:.4f} , {:.4f}.".format(f1 , acc)
