import pandas as pd
import numpy as np
from sklearn import preprocessing

#Label encodes the numerical data


def main():
	priceLabels = pd.read_csv('priceLabels.csv')
	le = preprocessing.LabelEncoder()
	prices = le.fit_transform(priceLabels['Forward Price'])

	priceLabels = priceLabels.drop(columns='Forward Price')
	priceLabels['Forward Price'] = prices
	print(priceLabels)

	priceLabels.to_csv('priceLabelsEncoded.csv', index = False)

if __name__ == "__main__":
	main()