import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
#from sklearn import cross_validation, linear_model
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import statistics
from sklearn.ensemble import StackingRegressor



#Models the data
def model(xTrain, yTrain, xTest, yTest):
    kfold = StratifiedKFold(n_splits=3)

    random_state = 3
    classifiers = []
    classifiers.append(SVR(C=1.0, epsilon=0.2))
    classifiers.append(DecisionTreeRegressor(random_state=random_state))
    classifiers.append(RandomForestRegressor(random_state=random_state, n_estimators=100))
    classifiers.append(GradientBoostingRegressor(random_state=random_state))
    classifiers.append(KNeighborsRegressor())

    cv_results = []
    for classifier in classifiers:
        cv_results.append(cross_validate(classifier, xTrain, yTrain, scoring = 'neg_mean_squared_error', cv = 3))


    
    cv_means = []
    cv_std = []
    for cv_result in cv_results:
        cv_means.append(abs(statistics.mean(cv_result['test_score'])))
        cv_std.append(statistics.stdev(cv_result['test_score']))

    cv_res = pd.DataFrame({"Mean_Squared_Errors":cv_means,"Algorithm":["SVR","Decision Tree", "RandomForest","GradientBoosting","KNeighboors"]})
    
    #print(cv_res)
    
    cv_res.plot(kind='bar',x='Algorithm',y='Mean_Squared_Errors')
    plt.show()
    
    '''
    # Working with BEST
    
    classifiersBest = []
    
    
    bestRFR = RandomForestRegressor(random_state=random_state, max_depth = 10, max_features = 25, min_samples_leaf = 1, n_estimators=100)
    
    bestDTR = DecisionTreeRegressor(random_state=random_state, max_depth = 50, max_features = 25, min_samples_leaf = 1)
    
    bestGBR = GradientBoostingRegressor(random_state=random_state, learning_rate = 0.01, max_depth = 3, max_features = 25, min_samples_leaf = 5)
    
    
    
    # SHOULD GO HERE
    
    bestRFR.fit(xTrain,yTrain)
    bestDTR.fit(xTrain,yTrain)
    bestGBR.fit(xTrain,yTrain)
    
    # added here
    
    classifiersBest.append(bestRFR)
    classifiersBest.append(bestDTR)
    classifiersBest.append(bestGBR)
    
    
    # stacking
    
    estimators = [('RFR', bestRFR), ('DTR', bestDTR), ('GBR', bestGBR)]
    
    reg = StackingRegressor(estimators = estimators, final_estimator=RandomForestRegressor(random_state=random_state, n_estimators=10))
    
    classifiersBest.append(reg)
    
    cv_results_best = []
    for classifier in classifiersBest:
        cv_results_best.append(cross_validate(classifier, xTrain, yTrain, scoring = 'neg_mean_squared_error', cv = 3))
        
    cv_means_best = []
    cv_std_best = []
    for cv_result in cv_results_best:
        cv_means_best.append(abs(statistics.mean(cv_result['test_score'])))
        cv_std_best.append(statistics.stdev(cv_result['test_score']))

    cv_res_best = pd.DataFrame({"Mean_Squared_Errors":cv_means_best,"Algorithm":["RF","DT", "GB", "Ensemble"]})
    
    #print(cv_res)
    
    cv_res_best.plot(kind='bar',x='Algorithm',y='Mean_Squared_Errors')
    plt.show()
    
    # UNTIL HERE
    
    '''
    
    #Stacking using optimized models and sklearn
    '''
    bestRFR.fit(xTrain,yTrain)
    bestDTR.fit(xTrain,yTrain)
    bestGBR.fit(xTrain,yTrain)
    '''
    
    
    
    #print("Ensemble Score: ", reg.fit(xTrain, yTrain).score(xTest, yTest))
    
    
    '''

    #Optimize support vector machine and predict
    SVM = SVC(probability=True)
    svc_grid = {'gamma': [ 0.001, 0.01, 0.1, 1],
                      'C': [1, 10, 50, 100, 250]}

    gsSVM = GridSearchCV(SVM,param_grid = svc_grid, cv=kfold, n_jobs = -1, scoring="accuracy", verbose = 1)
    gsSVM.fit(xTrain,yTrain)
    bestSVM = gsSVM.best_estimator_
    print(bestSVM.get_params())

    yHat = bestSVM.predict(xTest)
    fpr, tpr, _ = roc_curve(yTest, yHat)
    plt.plot(fpr, tpr, label="SVM")
    print('SVM Accuracy Score: ' + str(accuracy_score(yHat, yTest)))
    
    '''
    
    scoreArr = []
    
    #Optimize random forest and predict
    RFR = RandomForestRegressor()
    rf_grid = {"max_depth": [1,3,5,10, 20, 50],
              "max_features": [5, 10, 20, 25],
              "min_samples_leaf": [1, 5, 10],
              "n_estimators" :[100,250,500],}

    gsRFR = GridSearchCV(RFR, param_grid = rf_grid, cv=3, n_jobs = -1, scoring="neg_mean_squared_error", verbose = 2)
    gsRFR.fit(xTrain,yTrain)
    print(gsRFR.best_params_)

    #yHat = bestRFR.predict(xTest)
    bestRFR = gsRFR.best_estimator_
    grid_accuracy = evaluate(bestRFR, xTest, yTest)
    
    print("Accuracy: ", grid_accuracy)
    scoreArr.append(grid_accuracy)
    
    
    #Optimize decision tree and predict
    KNN = KNeighborsRegressor()
    knn_grid = {"n_neighbors": [1,3,5,10, 15, 20]}

    gsKNN = GridSearchCV(KNN, param_grid = knn_grid, cv=3, n_jobs = -1, scoring="neg_mean_squared_error", verbose = 2)
    gsKNN.fit(xTrain,yTrain)
    #bestDTR = gsDTR.best_params_
    print(gsKNN.best_params_)

    #yHat = bestDTR.predict(xTest)
    bestKNN = gsKNN.best_estimator_
    grid_accuracy = evaluate(bestKNN, xTest, yTest)
    
    print("Accuracy: ", grid_accuracy)
    scoreArr.append(grid_accuracy)
    
    
    #Optimize gradient boosting and predict
    GBR = GradientBoostingRegressor()
    gbr_grid = {"learning_rate": [0.01,0.05,0.1,0.2, 0.3],
              "max_features": [5, 10, 20, 25],
              "min_samples_leaf": [1, 5, 10],
               "max_depth": [1, 3, 5]}

    gsGBR = GridSearchCV(GBR, param_grid = gbr_grid, cv=3, n_jobs = -1, scoring="neg_mean_squared_error", verbose = 2)
    gsGBR.fit(xTrain,yTrain)
    #bestGBR = gsGBR.best_params_
    print(gsGBR.best_params_)

    #yHat = bestDTR.predict(xTest)
    bestGBR = gsGBR.best_estimator_
    grid_accuracy = evaluate(bestGBR, xTest, yTest)
    
    print("Accuracy: ", grid_accuracy)
    scoreArr.append(grid_accuracy)
    
    
    
    
    
    # Stacking Ensemble
    
    estimators = [('RFR', bestRFR), ('KNN', bestKNN), ('GBR', bestGBR)]

    #Stacking using optimized models and sklearn
    reg = StackingRegressor(estimators = estimators, final_estimator=RandomForestRegressor(random_state=random_state, n_estimators=10))
    
    for x in scoreArr:
        print("Accuracy: ", x)
    
    print("Ensemble Score: ", reg.fit(xTrain, yTrain).score(xTest, yTest))
    
    
    
    
    return None

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
    
    '''
    fpr, tpr, _ = roc_curve(yTest, yHat)
    plt.plot(fpr, tpr, label="Random Forest")
    print('Random Forest Accuracy Score: ' + str(accuracy_score(yHat, yTest)))
    '''
    
    
    '''

    #Logistic Regression Prediction
    LR = LogisticRegression()
    LR.fit(xTrain,yTrain)
    yHat = LR.predict(xTest)
    fpr, tpr, _ = roc_curve(yTest, yHat)
    plt.plot(fpr, tpr, label="Logistic Regression")
    print('LogisticRegression Accuracy Score: ' + str(accuracy_score(yHat, yTest)))

    estimators = [('SVM',bestSVM), ('RFC', bestRFC), ('LR', LR)]

    #Stacking using optimized models and sklearn
    clf = StackingClassifier(estimators= estimators, final_estimator=LogisticRegression())
    clf.fit(xTrain,yTrain)
    yHat = clf.predict(xTest)
    fpr, tpr, _ = roc_curve(yTest, yHat)
    plt.plot(fpr, tpr, label="Stack Ensemble")
    print('Stack Accuracy Score: ' + str(accuracy_score(yHat, yTest)))


    #Plots guessing dominant class
    yHat = [1] * len(xTest)
    fpr, tpr, _ = roc_curve(yTest, yHat)
    plt.plot(fpr, tpr, '--', label="Base Line")

    #Labels and plotting
    sns.set_style("whitegrid")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Encoded Approach')
    plt.legend()
    plt.show()
    '''

    
         
          
#Creates Explained Variance Graph
def myPCA(xTrain, xTest):
	pca = PCA(.98).fit(xTrain.values)
	newTrain = pca.transform(xTrain)
	newTest = pca.transform(xTest)
	
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.xlabel('number of components')
	plt.ylabel('cumulative explained variance');
	plt.title('Explained Variance Across Components')
	print(pca.explained_variance_ratio_)
	plt.show()
	
	return newTrain, newTest

#Creates correlation heatmap
def heatmap(xTrain, yTrain, xTest):
	xTrain['Forward Price'] = yTrain
	cor = xTrain.corr()
	cor_target = abs(cor['Forward Price'])
	print(sorted(cor_target, reverse = True))
	plt.show()

#Standard-scaler the data
def scale(xTrain, xTest):
	scaler = StandardScaler().fit(xTrain)
	scaledTrain = scaler.transform(xTrain.values)
	scaledTest = scaler.transform(xTest.values)

	xNewTrain = pd.DataFrame(scaledTrain, index=xTrain.index, columns=xTrain.columns)
	xNewTest = pd.DataFrame(scaledTest, index=xTest.index, columns=xTest.columns)

	return xTrain, xTest

def main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain", help="filename for features of the training data")
    parser.add_argument("yTrain", help="filename for labels associated with training data")
    parser.add_argument("xTest", help="filename for features of the test data")
    parser.add_argument("yTest", help="filename for labels associated with the test data")

    args = parser.parse_args()
    '''
    xTrain = pd.read_csv('xTrainPricesPCA.csv')
    yTrain = pd.read_csv('yTrainPrices.csv')
    xTest = pd.read_csv('xTestPricesPCA.csv')
    yTest = pd.read_csv('yTestPrices.csv')
    
    yTrain = yTrain.values.ravel()
    yTest = yTest.values.ravel()
    
    '''
    lab_enc = preprocessing.LabelEncoder()
    yTrainArray = lab_enc.fit_transform(yTrain)
    yTestArray = lab_enc.fit_transform(yTest)
    '''

    sns.set(style='white', context='poster', palette='pastel')

    xTrain, xTest = scale(xTrain, xTest)

    #xTrain, xTest = myPCA(xTrain, xTest)
    #heatmap(xTrain, yTrain, xTest)
    model(xTrain, yTrain, xTest, yTest)



if __name__ == "__main__":
	main()