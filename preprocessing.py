import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import statistics
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
import heapq


def scale(xTrain, xTest):
    scaler = StandardScaler().fit(xTrain)
    scaledTrain = scaler.transform(xTrain.values)
    scaledTest = scaler.transform(xTest.values)

    xScaledTrain = pd.DataFrame(scaledTrain, index=xTrain.index, columns=xTrain.columns)
    xScaledTest = pd.DataFrame(scaledTest, index=xTest.index, columns=xTest.columns)

    return xScaledTrain, xScaledTest

def correlation(xTrain, yTrain, xTest, yTest):
    xTrain['strategy'] = yTrain
    xTest['strategy'] = yTest
    cor = xTrain.corr()
    cor_target = abs(cor['strategy'])
    median = (statistics.median(cor_target))
    
    # remove bottom half of correlated objects
    relevant_features = cor_target[cor_target>median]
    
    
    
    # find the most important features
    xTrain = xTrain[relevant_features.index]
    xTest = xTest[relevant_features.index]
    xTrain = xTrain.drop(columns=['strategy'])
    xTest = xTest.drop(columns=['strategy'])
    
    # sort relevant_features and print top five features
    relevant_features = relevant_features.sort_values(ascending=False)
    #print(relevant_features[:6])
    
    return xTrain, xTest

def lassoReg(xTrain, yTrain, xTest):
    scaler = StandardScaler()
    scaler.fit(xTrain)
    #scaler.fit(xTrain.fillna(0))
    select = SelectFromModel(LogisticRegression(C=1, penalty='l1', max_iter=200, solver = 'liblinear'))
    select.fit(xTrain, yTrain)
    
    selected_feat = xTrain.columns[(select.get_support())]
    #removed_feats = xTrain.columns[(select.estimator_.coef_ == 0).ravel().tolist()]
    
    
    # drop the removed_feats
    xTrain = select.transform(xTrain.fillna(0))
    xTest = select.transform(xTest.fillna(0))
    xTrain = pd.DataFrame(xTrain, columns = selected_feat)
    xTest = pd.DataFrame(xTest, columns = selected_feat)
    
    #xTrain = xTrain.drop(columns=removed_feats)
    #xTest = xTest.drop(columns=removed_feats)
    
    return xTrain, xTest


def heatmap(xTrain):
    
    # plot a heatmap
    plt.figure(figsize=(12,10))
    xTrain = pd.DataFrame(xTrain)
    cor = xTrain.corr()
    sns.heatmap(cor, cmap=plt.cm.Reds)
    plt.show()
    
    return None


def pcaPlot(xTrain, xTest):
    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)
    # pca analysis with excessive amount of components for graphing purposes
    pca = PCA(n_components = 30).fit(xTrain.values)
    #newTrain = pca.transform(xTrain)
    #newTest = pca.transform(xTest)
    
    #Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('Explained Variance Across Components')
    plt.show()
    
    return None

def pcaAnalysis(xTrain, xTest):
    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)
    # conduct PCA Analysis
    pca = PCA(n_components = 0.95, svd_solver = 'full')
    pca.fit(xTrain.values)
    newTrain = pca.transform(xTrain)
    newTest = pca.transform(xTest)
    
    # number of components
    numberComponents = pca.components_.shape[0]
    print("Number of components for 95% explained variance: ", numberComponents)
    
    # find most important components
    for i1 in range(3):
        print("Most important features: ", heapq.nlargest(3, range(len(pca.components_[i1])), pca.components_[i1].take))
    
    
    return newTrain, newTest

def main():
    sns.set(style='white', context='notebook', palette='pastel')
    
    # import data
    xTrain = pd.read_csv('xTrain.csv')
    xTest = pd.read_csv('xTest.csv')
    yTrainPrices = pd.read_csv('yTrainPrices.csv')
    yTestPrices = pd.read_csv('yTestPrices.csv')
    yTrainBinary = pd.read_csv('yTrainBinary.csv')
    yTestBinary = pd.read_csv('yTestBinary.csv')

    # scale data
    xTrain, xTest = scale(xTrain, xTest)
    
    # conduct correlation analysis
    # prices
    #print("Correlation with Prices")
    xTrainPrices, xTestPrices = correlation(xTrain, yTrainPrices, xTest, yTestPrices)
    # binary
    #print("Correlation with Binary")
    xTrainBinary, xTestBinary = correlation(xTrain, yTrainBinary, xTest, yTestBinary)
    
    # conduct lasso reg
    # prices
    xTrainPrices, xTestPrices = lassoReg(xTrainPrices, yTrainPrices, xTestPrices)
    # binary
    xTrainBinary, xTestBinary = lassoReg(xTrainBinary, yTrainBinary, xTestBinary)
    
    # make Pearson Heatmap
    # prices
    heatmap(xTrainPrices)
    # binary
    heatmap(xTrainBinary)
    
    # PCA Plot
    # prices
    pcaPlot(xTrainPrices, xTestPrices)
    # binary
    pcaPlot(xTrainBinary, xTestBinary)
    
    # PCA Analysis
    # prices
    print("Prices PCA: ")
    xTrainPricesPCA, xTestPricesPCA = pcaAnalysis(xTrainPrices, xTestPrices)
    # binary
    print("Binary PCA: ")
    xTrainBinaryPCA, xTestBinaryPCA = pcaAnalysis(xTrainBinary, xTestBinary)
    
    
    
    # write CSV Files
    xTrainPrices = pd.DataFrame(xTrainPrices)
    xTestPrices = pd.DataFrame(xTestPrices)
    xTrainBinary = pd.DataFrame(xTrainBinary)
    xTestBinary = pd.DataFrame(xTestBinary)
    xTrainPricesPCA = pd.DataFrame(xTrainPricesPCA)
    xTestPricesPCA = pd.DataFrame(xTestPricesPCA)
    xTrainBinaryPCA = pd.DataFrame(xTrainBinaryPCA)
    xTestBinaryPCA = pd.DataFrame(xTestBinaryPCA)
    
    # without PCA Analysis
    xTrainPrices.to_csv('xTrainPrices.csv', index = False)
    xTestPrices.to_csv('xTestPrices.csv', index = False)
    xTrainBinary.to_csv('xTrainBinary.csv', index = False)
    xTestBinary.to_csv('xTestBinary.csv', index = False)
    
    # with PCA Analysis
    xTrainPricesPCA.to_csv('xTrainPricesPCA.csv', index = False)
    xTestPricesPCA.to_csv('xTestPricesPCA.csv', index = False)
    xTrainBinaryPCA.to_csv('xTrainBinaryPCA.csv', index = False)
    xTestBinaryPCA.to_csv('xTestBinaryPCA.csv', index = False)
    



if __name__ == "__main__":
	main()