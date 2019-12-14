import pandas as pd
import numpy as np

#Removes columns that are missing many values while imputing the mean for other missing values

def main():
	xTrain = pd.read_csv('featureEngineeredData.csv')
	xTrain = xTrain.replace(np.inf, None)
	xTrain = xTrain.replace(np.NINF, None)
	xTrain = xTrain.replace(np.nan, None)
	xTrain = xTrain.replace('#NAME?', None)

	for col in xTrain:
		if int(xTrain[col].isnull().sum()) / len(xTrain) > 0.10:
			xTrain = xTrain.drop(columns = [col])
		elif int(xTrain[col].isnull().sum()) > 0:
			xTrain[col].fillna(xTrain[col].mean(), inplace = True)

	xTrain.to_csv('imputedData.csv', index = False)
	

if __name__ == "__main__":
	main()