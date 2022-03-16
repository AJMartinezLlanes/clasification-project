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

# Question 1 visualization
def churn_rate():
    print(f'Churn percentage {round(train.churn_encoded.mean(),4)*100}%')
    y = train.churn.value_counts()
    plt.pie(y,labels=['No Churn','Churn'] ,explode= [0,0.2], autopct='%.2f%%')
    plt.show()

# Heat map
def churn_heat_map():
    plt.figure(figsize=(18,22))
    churn_heatmap = sns.heatmap(train.corr()[['churn_encoded']].sort_values(by='churn_encoded', ascending=False), vmin=-.5, vmax=.5, annot=True)
    churn_heatmap.set_title('Features Correlating with Churn')

# Question 2 Stats
def top_3_churn():
    print('Contract Type vs Churn')
    print('HO: There is no relation between month-to-month contract and churn rate')
    print('H⍺: There is a relation between month-to-month contract and churn rate')
    print('---------------------------------------------\n')
    observed = pd.crosstab(train.contract_type , train.churn_encoded)
    print(observed)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n:')
    print(observed.values)
    print('------------------------\nExpected: \n')
    print(expected.astype(int))
    print('------------------------\n')
    print(f'chi2 = {chi2:.2f}')
    print(f'p value: {p:.4f}')
    print(f'degrees of freedom: {degf}')
    if (p < alpha):
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

# Question 2 visualizations
def top_3_viz():
    plt.title("Churn vs Contract Type")
    sns.barplot(x="contract_type", y="churn_encoded", data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r", label = "Churn avg")
    plt.xlabel('')
    plt.legend()
    plt.show()
    plt.title("Churn vs Online Security")
    sns.barplot(x="online_security", y="churn_encoded", data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r", label="Churn avg")
    plt.xlabel('')
    plt.legend()
    plt.show()
    plt.title("Churn vs Tech Support")
    sns.barplot(x="tech_support", y="churn_encoded", data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r", label="Churn avg")
    plt.xlabel('')
    plt.legend()
    plt.show()
# Qeu
def q_3_viz():
    no_int_cols = [col for col in train if col.endswith('no_internet_service')]
    for col in no_int_cols:
        plt.title(f"Churn vs {col}")
        sns.barplot(x=f'{col}', y="churn_encoded", data=train)
        churn_rate = train.churn_encoded.mean()
        plt.axhline(churn_rate, color="r", label="Churn avg")
        plt.xlabel('')
        plt.legend()
        plt.show()


# Question 4 Stats
def q_4_churn():
    print('Senior Citizen vs Churn')
    print('HO: There is no relation between month-to-month senior citizen and churn rate')
    print('H⍺: There is a relation between month-to-month senior citizen and churn rate')
    print('---------------------------------------------\n')
    observed = pd.crosstab(train.senior_citizen, train.churn_encoded)
    print(observed)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n:')
    print(observed.values)
    print('------------------------\nExpected: \n')
    print(expected.astype(int))
    print('------------------------\n')
    print(f'chi2 = {chi2:.2f}')
    print(f'p value: {p:.4f}')
    print(f'degrees of freedom: {degf}')
    if (p < alpha):
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

# Question 4 visualizations
def q_4_viz():
    plt.title("Churn vs Senior Citizen")
    sns.barplot(x="senior_citizen", y="churn_encoded", hue='gender', data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r", label="Churn avg")
    plt.legend()
    plt.show()
    plt.title("Churn vs Senior Citizen")
    sns.barplot(x="senior_citizen", y="payment_type", hue='gender' ,data=train)
    plt.show()
    plt.title("Churn vs Gender")
    sns.barplot(x="gender", y="churn_encoded",hue="senior_citizen", data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r",label="Churn avg")
    plt.legend()
    plt.show()
    plt.title("Churn vs Dependents")
    sns.barplot(x="dependents", y="churn_encoded", hue="gender",data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r",label="Churn avg")
    plt.legend()
    plt.show()
    plt.title("Churn vs Partner")
    sns.barplot(x="partner", y="churn_encoded",hue="gender", data=train)
    churn_rate = train.churn_encoded.mean()
    plt.axhline(churn_rate, color="r", label="Churn avg")
    plt.legend()
    plt.show()

