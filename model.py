import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn import tree
from sklearn.svm import SVC
from scipy.stats import pearsonr as pearson
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, confusion_matrix

seed = 212

def getInputFeatures():
    inputFeatures  =  [0.38,	636.64,	18,	1.26,	3.56	,21,	0.43, 27,	5,	9.23
]
    return inputFeatures


def train(data, model):
    X = np.array([data['ttr'], data['R'], data['num_concepts_mentioned'], 
                data['ARI'], data['CLI'], data['prp_count'], data['prp_noun_ratio'],data['NP_count'], data['VP_count'], 
                data['word_sentence_ratio']])
    
    X = X.T
    y = np.array(data['Category']).T
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=212)
    train_samples, n_features = X.shape
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X_train, y_train):
        # Train the model

        if type(model).__name__ == 'RandomForestClassifier':
            model.n_estimators = 10
        
        model.fit(X_train[train], y_train[train])    
        
        # evaluate the model
        y_pred = model.predict(X_train[test])
        
        # evaluate predictions
        cvscores.append(accuracy_score(y_train[test], y_pred))
        
    # print(np.mean(cvscores))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(type(model).__name__)
    
    print(model.predict_proba(X_test).mean(axis=0))
    print(model.predict_proba([usedFeatures]))
    
    print('Test accuracy for model: {}'.format(accuracy))

    # finalAns[str(type(model).__name__)] = {
    #     'category' : int(model.predict([usedFeatures])[0]),
    #     'accuracy' :float(accuracy),
    #     'probabilty_0' :float (model.predict_proba([usedFeatures])[0][0]),
    #     'probabilty_1' : float(model.predict_proba([usedFeatures])[0][1])
    # } 

    finalAns[str(type(model).__name__) + '_category'] = int(model.predict([usedFeatures])[0])
    finalAns[str(type(model).__name__) + '_accuracy'] = float(accuracy)
    finalAns[str(type(model).__name__) + '_probability_0'] = float (model.predict_proba([usedFeatures])[0][0])
    finalAns[str(type(model).__name__) + '_probability_1'] = float(model.predict_proba([usedFeatures])[0][1])

    # print ('F1-score: {}'.format(f1_score(y_test, y_pred, average=None)))
    # print (classification_report(y_test, y_pred))
  
        
def main(usedFeatures2):
    print('IN MODEL')
    global usedFeatures
    usedFeatures = usedFeatures2
    global finalAns 
    finalAns = {}

    
    # Read data
    data = pd.read_csv(os.path.join(os.getcwd(),'feature_set_dem.csv'), encoding='utf-8')

    models = [LogisticRegression(), 
    tree.DecisionTreeClassifier(), 
    # SVC(probability=True), 
    RandomForestClassifier()]

    for mod in models:    
        train(data, mod)
        print(mod.predict([usedFeatures]))

        if(mod.predict([usedFeatures])):
            print('Dementia')
        else:
            print('Control')
        print('\n')


    print(finalAns)

    return finalAns

if __name__ == '__main__':
    main(usedFeatures2=getInputFeatures())