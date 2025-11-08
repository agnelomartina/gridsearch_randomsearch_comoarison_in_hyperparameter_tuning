**why do we need hyperparameter tuning in a decisiontree classifier?**
  If a decisiontree is too deep,it overfits(memorises training data) and if its too shallow,it underfits(misses patterns).
  Hyperparameter tuning finds the sweet spot by making a right balance between bias and variance.
  
**how can we achieve hyperparameter tuning?**
  By means of GridSearch and RandomSearch

**GridSearch:**
  It tests every possible combination of hyper parameter values that is specified in a grid
  **Library**:from sklearn.model_selection import GridSearchCV
  
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

**RandomSearch:**
  It picks the random combination of hyperparameters from the defined ranges
  **Library**:from sklearn.model_selection import RandomizedSearchCV
  
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

**comparison of the result of GridSearch and RandomSearch**
----------Grid Search-------------
Best hyperparameters:  {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
Best cross-validation accuracy:  0.6917935483870968
Test Accuracy: 0.6828358208955224
-----------Random Search--------------
Best hyperparameters: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_leaf': 2, 'min_samples_split': 10}
Best cross-validation accuracy: 0.6949935483870968
Test Accuracy: 0.6940298507462687
