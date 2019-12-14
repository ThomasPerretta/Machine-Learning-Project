import pandas as pd
import numpy as np

#Classifies the data to either buy or sell the stock based on if the price is higher in 3 months

def main():
	xData = pd.read_csv('xTrain.csv')
	yData = pd.read_csv('priceLabels.csv')
	trainBinary = []

	
	for index in range(len(xData)):
		if float(xData.iloc[index]['Current Price']) <= float(yData.iloc[index]['Forward Price']):
			trainBinary.append(1)
		else:
			trainBinary.append(0)


	df = pd.DataFrame()
	df['Strategy'] = trainBinary

	df.to_csv('binaryLabels.csv', index = False)


if __name__ == "__main__":
	main()