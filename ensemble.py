import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc



#Models the data
def model(xTrain, yTrain, xTest, yTest):
	kfold = KFold(n_splits=10)

	#Models tested
	classifiers = []
	classifiers.append(SVC(gamma = 'auto'))
	classifiers.append(DecisionTreeClassifier())
	classifiers.append(RandomForestClassifier(n_estimators = 100))
	classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 100))
	classifiers.append(GradientBoostingClassifier())
	classifiers.append(LogisticRegression(solver = 'lbfgs'))
	classifiers.append(KNeighborsClassifier())
	classifiers.append(GaussianNB())


	#See which models perform the best before long optimization process
	crossVal_results = []
	for c in classifiers:
		crossVal_results.append(cross_val_score(c, xTrain, yTrain['Strategy'], scoring = 'accuracy', cv = kfold))

	means = []
	std = []
	for c in crossVal_results:
		means.append(c.mean())
		std.append(c.std())

	cv_res = pd.DataFrame({'CrossValMeans':means,'CrossValerrors': std,'Algorithm':['SVM','Decision Tree',
	'Random Forest', 'AdaBoost', 'Gradient Boosting','Logistic Regression','K-Neighbors', 'Naive Bayes']})

	p = sns.barplot("CrossValMeans","Algorithm",data = cv_res)
	p.set_xlabel("Mean Accuracy")
	p = p.set_title("Cross Validation Scores for Binary Approach with PCA")
	
	plt.show()

	#Optimize support vector machine and predict
	SVM = SVC(probability=True)
	svc_grid = {'gamma': [ 0.001, 0.01, 0.1, 1],
                  	'C': [1, 10, 50, 100, 250]}

	gsSVM = GridSearchCV(SVM,param_grid = svc_grid, cv=kfold, n_jobs = -1, scoring="accuracy", verbose = 1)
	gsSVM.fit(xTrain,yTrain['Strategy'])
	bestSVM = gsSVM.best_estimator_
	print(bestSVM.get_params())

	yHat = bestSVM.predict(xTest)
	fpr, tpr, _ = roc_curve(yTest, yHat)
	plt.plot(fpr, tpr, label="SVM")
	print('SVM Accuracy Score: ' + str(accuracy_score(yHat, yTest)))
	
	#Optimize random forest and predict
	RFC = RandomForestClassifier()
	rf_grid = {"max_depth": [1,3,5,10, 20, 50],
              "max_features": [5, 10, 20, 25],
              "min_samples_leaf": [1, 5, 10],
              "n_estimators" :[100,250,500],}

	gsRFC = GridSearchCV(RFC,param_grid = rf_grid, cv=kfold, n_jobs = -1, scoring="accuracy", verbose = 1)
	gsRFC.fit(xTrain,yTrain['Strategy'])
	bestRFC = gsRFC.best_estimator_
	print(bestRFC.get_params())

	yHat = bestRFC.predict(xTest)
	fpr, tpr, _ = roc_curve(yTest, yHat)
	plt.plot(fpr, tpr, label="Random Forest")
	print('Random Forest Accuracy Score: ' + str(accuracy_score(yHat, yTest)))

	#Logistic Regression Prediction
	LR = LogisticRegression()
	LR.fit(xTrain,yTrain['Strategy'])
	yHat = LR.predict(xTest)
	fpr, tpr, _ = roc_curve(yTest, yHat)
	plt.plot(fpr, tpr, label="Logistic Regression")
	print('LogisticRegression Accuracy Score: ' + str(accuracy_score(yHat, yTest)))

	estimators = [('SVM',bestSVM), ('RFC', bestRFC), ('LR', LR)]

	#Stacking using optimized models and sklearn
	clf = StackingClassifier(estimators= estimators, final_estimator=LogisticRegression())
	clf.fit(xTrain,yTrain['Strategy'])
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
	plt.title('ROC Curve for Binary Approach with PCA')
	plt.legend()
	plt.show()

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
	xTrain['Strategy'] = yTrain
	cor = xTrain.corr()
	cor_target = abs(cor['Strategy'])
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
	parser = argparse.ArgumentParser()
	parser.add_argument("xTrain", help="filename for features of the training data")
	parser.add_argument("yTrain", help="filename for labels associated with training data")
	parser.add_argument("xTest", help="filename for features of the test data")
	parser.add_argument("yTest", help="filename for labels associated with the test data")

	args = parser.parse_args()
	xTrain = pd.read_csv(args.xTrain)
	yTrain = pd.read_csv(args.yTrain)
	xTest = pd.read_csv(args.xTest)
	yTest = pd.read_csv(args.yTest)

	sns.set(style='white', context='poster', palette='pastel')

	xTrain, xTest = scale(xTrain, xTest)

	#xTrain, xTest = myPCA(xTrain, xTest)
	#heatmap(xTrain, yTrain, xTest)
	model(xTrain, yTrain, xTest, yTest)



if __name__ == "__main__":
	main()