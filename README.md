# Kaggle - Digit Recognizer

 Computer vision techniques to identify digits from a dataset of tens of thousands of handwritten images. <br />
 https://www.kaggle.com/c/digit-recognizer
 
# Outcomes

Performance measured as the accuracy on validation data per model: <br />
(The below values are indicative in the sense that they highly depend on the selection of parameters such as the PCA component range, the seed in KNN, the number of estimators in Random Forest and many others)

1. SVM with PCA : 97.9% <br />
2. KNN with PCA : 97.6% <br />
3. XGBoost with parameter tuning : 96.2% <br />
4. Random Forest Classifier : 88.7% <br />
5. Multiple Linear Regression : 85.1% <br />

# Data Preprocessing

 • Load Data <br />
 • Check for null/missing values <br />
 • Check for unbalanced labels <br />
 • Data normalization <br />
 • Label encoding (One Hot Encoding to convert categorical variables to one hot vectors) <br />
 • Split training and validation sets <br />
 
# Training models

 • Multiple Linear Regression <br />
 • Support Vector Machine (SVM) with Principal Component Analysis (PCA) <br />
 • eXtreme Gradient Boosting (XGBoost) with parameter tuning <br />
 • Random Forest Classifier <br />
 • K Nearest Neighbors Classifier (KNN) with Principal Component Analysis (PCA) <br />
 
# Evaluating models
 
 Evaluation performed based on both the F1 score and the deduced accuracy of each model on the validation data.
