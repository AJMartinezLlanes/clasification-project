# import the usual libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# import skleand funtions needed for this project 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, f1_score

#Turn off warning 
import warnings
warnings.filterwarnings("ignore")

#import local libraries
import prepare
import acquire
import env

alpha = .05

df = acquire.get_telco_data()
df = prepare.prep_telco(df)

train, validate, test = prepare.split_telco_data(df)
train.shape, validate.shape, test.shape

# modeling variables

obj_cols = df.columns[[df[col].dtype == 'O' for col in df.columns]]
df.drop(columns= obj_cols, inplace=True)
train, validate, test = prepare.split_telco_data(df)


x_train = train.drop(columns=['churn_encoded'])
y_train = train.churn_encoded

x_validate = validate.drop(columns=['churn_encoded'])
y_validate = validate.churn_encoded

x_test = test.drop(columns=['churn_encoded'])
y_test = test.churn_encoded

# modeling
def baseline_churn():
    baseline = y_train.mode()[0]
    baseline_accuracy = (train.churn_encoded==0).mean()
    print(f'Baseline accuracy: {baseline_accuracy:.4%}')

# decision tree function
def dec_tree_class():
    metrics = []
    for i in range(1, 16):
        tree = DecisionTreeClassifier(max_depth=i, random_state=177)
        tree = tree.fit(x_train, y_train)
        in_sample_accuracy = tree.score(x_train, y_train)
        out_of_sample_accuracy = tree.score(x_validate, y_validate)
        output = {
            "max_depth": i,
            "train_accuracy": in_sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }
        metrics.append(output)
 
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    print('Max depth 6 performs the best')
    print(df)

# Random Tree Function
def ran_tree_class():
    metrics = []
    max_depth = 10

    for i in range(1, max_depth):
        # Make the thing
        depth = max_depth -i
        n_samples = i
        forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=n_samples, random_state=1349)

        # Fit the thing
        forest = forest.fit(x_train, y_train)

        # Use the thing
        sample_accuracy = forest.score(x_train, y_train)
        
        out_of_sample_accuracy = forest.score(x_validate, y_validate)

        output = {
            "min_per_leaf": n_samples,
            "max_depth": depth,
            "train_accuracy": sample_accuracy,
            "validate_accuracy": out_of_sample_accuracy
        }
        
        metrics.append(output)
        
    df = pd.DataFrame(metrics)
    df["difference"] = df.train_accuracy - df.validate_accuracy
    print('Max depth 5, Min leaf 5 performs the best')
    print(df)

# KNN Model
def knn_model():
    print('KNC with 20 neighbors performs the best')
    knc = KNeighborsClassifier(20)
    knc.fit(x_train, y_train)
    y_predictions = knc.predict(x_train)

    mod_score = knc.score(x_train, y_train)
    con_matrix = pd.DataFrame(confusion_matrix(y_train, y_predictions))
    class_report = classification_report(y_train, y_predictions)

    print(f'Accuracy using Model Score: {mod_score:.2%}')
    print(f'Using Confusion Matrix:\n{con_matrix}')
    print(f'Class report:\n{class_report}')

    tn = con_matrix.loc[0,0]
    fn = con_matrix.loc[1,0]
    fp = con_matrix.loc[0,1]
    tp = con_matrix.loc[1,1]
    all = tp+fp+fn+tn
    print(f'True Positive(tp): {tp} \nFalse Positive(fp): {fp} \nFalse Negative(fn): {fn} \nTrue Negative(tn): {tn}')
    accuracy = (tp + tn)/all
    print(f"Accuracy: {accuracy:.4}")
    true_positive_rate = tp/(tp+fn)
    print(f"True Positive Rate: {true_positive_rate:.4}")
    false_positive_rate = fp/(fp+tn)
    print(f"False Positive Rate: {false_positive_rate:.4}")
    true_negative_rate = tn/(tn+fp)
    print(f"True Negative Rate: {true_negative_rate:.4}")
    false_negative_rate = fn/(fn+tp)
    print(f"False Negative Rate: {false_negative_rate:.4}")
    precision = tp/(tp+fp)
    print(f"Precision: {precision:.4}")
    recall = tp/(tp+fn)
    print(f"Recall: {recall:.4}")
    f1_score = 2*(precision*recall)/(precision+recall)
    print(f"F1 Score: {f1_score:.4}")
    support_pos = tp+fn
    print(f"Support (0): {support_pos}")
    support_neg = fp+tn
    print(f"Support (1): {support_neg}")