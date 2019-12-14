import pandas as pd
import numpy as np


#Splits the data into train and test splits based on dates

def main():
	fundamentals = pd.read_csv('imputedData.csv')
	priceLabels = pd.read_csv('priceLabelsEncoded.csv')
	stratLabels = pd.read_csv('binaryLabels.csv')

	fundamentals['Forward Price'] = priceLabels['Forward Price']
	fundamentals['Strategy'] = stratLabels['Strategy']

	train = []
	test = []
	
	for index, row in fundamentals.iterrows():
		if index+1 not in fundamentals.index:
			test.append(row)
		else:
			if row['Ticker Symbol'] == fundamentals.iloc[index+1]['Ticker Symbol']:
				train.append(row)
			else:
				test.append(row)



	xTrain = pd.DataFrame(train).reset_index(drop = True)
	xTest = pd.DataFrame(test).reset_index(drop = True)

	xTrain = xTrain.drop(columns = ['Ticker Symbol', 'Period Ending'])
	xTest = xTest.drop(columns = ['Ticker Symbol', 'Period Ending'])

	yTrainPrices = pd.DataFrame()
	yTrainPrices['Forward Price'] = xTrain['Forward Price']
	xTrain = xTrain.drop(columns = ['Forward Price'])


	yTestPrices = pd.DataFrame()
	yTestPrices['Forward Price'] = xTest['Forward Price']
	xTest = xTest.drop(columns = ['Forward Price'])

	yTrainBinary = pd.DataFrame()
	yTrainBinary['Strategy'] = xTrain['Strategy']
	xTrain = xTrain.drop(columns = ['Strategy'])


	yTestBinary = pd.DataFrame()
	yTestBinary['Strategy'] = xTest['Strategy']
	xTest = xTest.drop(columns = ['Strategy'])

	xTrain.to_csv('xTrain.csv', index = False)
	xTest.to_csv('xTest.csv', index = False)

	yTrainPrices.to_csv('yTrainPrices.csv', index = False)
	yTestPrices.to_csv('yTestPrices.csv', index = False)

	yTrainBinary.to_csv('yTrainBinary.csv', index = False)
	yTestBinary.to_csv('yTestBinary.csv', index = False)

if __name__ == "__main__":
	main()