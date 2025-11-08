import pandas as pd
from pandas.core.common import random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(df.columns)
print(df.head())

#preprocessing by labelencoder(converting categorical data into numerical data
le_object=LabelEncoder()
df['Cabin']=le_object.fit_transform(df['Cabin'])
print("label encoded values",df['Cabin'])

X=df[['PassengerId','Pclass','Age','Cabin']]
#X = df.iloc[:, [0,2,5,10]]--->choosing the no.of columns with index
Y = df['Survived']
print(X)
print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

my_model=DecisionTreeClassifier(random_state=42)

# Defining the hyperparameter tuning using GridSearch
my_param = {
    'criterion': ['gini', 'entropy'],  # Split criteria
    'max_depth': [ 3,5,7],   # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],   # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],     # Minimum samples required at a leaf node

}
# Initialize GridSearchCV with cross-validation and accuracy scoring
my_grid = GridSearchCV(estimator=my_model, param_grid=my_param, cv=5)

# Perform Grid Search to find the best hyperparameters
my_grid.fit(X_train, Y_train)

# Get the best parameters and the best score from the grid search
print("----------Grid Search-------------")
print("Best hyperparameters: ",my_grid.best_params_)
print("Best cross-validation accuracy: ",my_grid.best_score_)

# Evaluate the model with the best parameters on the test set
best_model = my_grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy:",accuracy_score(Y_test,y_pred))

#Defining hyperparameter tuning using RandomSearch
print("-----------Random Search--------------")
random_param={'criterion': ['gini', 'entropy'],  # Split criteria
    'max_depth': randint(3,5),   # Maximum depth of the tree
    'min_samples_split': randint(10,20),   # Minimum samples required to split a node
    'min_samples_leaf': randint(2,4)    # Minimum samples required at a leaf node
}
random_search=RandomizedSearchCV(estimator=my_model,param_distributions=random_param,n_iter=10,cv=5)

random_search.fit(X_train,Y_train)

# Get the best parameters and the best score from the random search
print("Best hyperparameters:", random_search.best_params_)
print("Best cross-validation accuracy:", random_search.best_score_)

# Evaluate the model with the best parameters on the test set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy:",accuracy_score(Y_test,y_pred))










