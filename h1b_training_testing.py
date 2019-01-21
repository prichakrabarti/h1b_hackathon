#Imports

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts

#Load the preprocessed file

df = pd.read_pickle("preprocessed_h1b.pkl")

#Setting up the feature matrix

X= df.drop(["case_status"],1)
X= pd.get_dummies(X)

#Setting up the target variable

y= df["case_status"]
le= LabelEncoder()
le.fit(y)
y= le.transform(y)

#Setting up our training and validation sets

X_train,X_test,y_train,y_test= tts(X,y,test_size=0.3,random_state=42)

#Training the model and tuning parameters using GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
params_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rfc_grid_search = GridSearchCV(estimator = rfc, 
	param_distributions = params_grid, cv = 5, random_state=42, n_jobs = -1)


#Training the model
rfc_grid_search.fit(X_train,y_train)

#Testing the model

y_pred= rfc_grid_search.predict(X_test)

from sklearn.metrics import roc_auc_score, accuracy_score

print ("The accuracy score of this model is: "+str(accuracy_score(y_test,y_pred))
print ("The roc_auc_score score of this model is: "+str(roc_auc_score(y_test,y_pred))	

	

