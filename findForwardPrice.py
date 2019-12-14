import pandas as pd

#Finds the forward price from the datasets given

def createLabels(fundamentals, prices):
	yLabels = []
	
	#Find 3-month forward prices
	stockPrices = prices.groupby('symbol') #Group by stock symbole

	for index, row in fundamentals.iterrows():
		stock = row['Ticker Symbol']
		date = row['Period Ending']
		originalDate = row['Period Ending']
		runs = 0

		found = False

		while not found: #Searches for a currentPrice for the row
			print(stock)
			runs+= 1
			currentStock = stockPrices.get_group(stock)
			for i, r in currentStock.iterrows(): #Ideally goes for the current price, looks for prices in the past if not
				if r['date'] == date:
					found = True
					if i+90 in currentStock.index:
						yLabels.append(float(stockPrices.get_group(stock).loc[i+90]['close']))
					else:
						cur = 1
						while not i+90-cur in currentStock.index:
							print('here')
							cur+=1

						yLabels.append(float(stockPrices.get_group(stock).loc[i+90-cur]['close']))


			splitDate = date.split('/')
			date = splitDate[0] + '/' + str(int(splitDate[1]) - 1) + '/' + splitDate[2]

			splitDate = date.split('/')
			if int(splitDate[1]) < 1:
				date = str(int(splitDate[0]) + 1) + '/' + '28' + '/' + splitDate[2]

			splitDate = date.split('/')
			if int(splitDate[0]) > 12:
				date = '1' + '/' + '28' + '/' + str(int(splitDate[2])+1)

	return yLabels


def main():
	# load the train and test data
	fundamentals = pd.read_csv('featureEngineeredData.csv')
	prices = pd.read_csv('prices-split-adjusted.csv')
	yLabels = createLabels(fundamentals, prices)

	df = pd.DataFrame()
	df['Forward Price'] = yLabels
	df.to_csv('priceLabels.csv', index = False)



if __name__ == "__main__":
	main()