import pandas as pd

#Finds the current price to add as a feature that is used by multiple ratios


def removeBadData(fundamentals):

	#Remove incomplete entries, those with less than 4 financial statements, removes 21 total
	for index, row in fundamentals.iterrows():
		if list(fundamentals['Ticker Symbol']).count(row['Ticker Symbol']) < 4:
			fundamentals = fundamentals.drop(index = index)

	fundamentals = fundamentals.drop(columns = ['Unnamed: 0'])

	#Remove bad-dates, entries with no price data
	for index, row in fundamentals.iterrows():
		date = row['Period Ending']
		splitDate = date.split('/')
		if int(splitDate[2]) < 2012 or int(splitDate[2]) > 2016:
			fundamentals = fundamentals.drop(index = index)


	#Remove entry with no price information, has fundamentals data but not price data
	for index, row in fundamentals.iterrows():
		if row['Ticker Symbol'] == 'UA':
			fundamentals = fundamentals.drop(index = index)

	return fundamentals


def findCurrentPrices(fundamentals):

	stockPrices = prices.groupby('symbol') #Group by stock symbole

	for index, row in fundamentals.iterrows():
		stock = row['Ticker Symbol']
		date = row['Period Ending']
		originalDate = row['Period Ending']
		runs = 0
		print(stock)
		found = False

		while not found: #Searches for a currentPrice for the row
			print(date)
			runs+= 1
			currentStock = stockPrices.get_group(stock)
			for i, r in currentStock.iterrows(): #Ideally goes for the current price, looks for prices in the past if not
				if r['date'] == date:
					found = True
					if i in currentStock.index:
						curPrice.append(float(stockPrices.get_group(stock).loc[i]['close']))
					else:
						cur = 1
						while not i-cur in currentStock.index:
							print('here')
							cur+=1

						curPrice.append(float(stockPrices.get_group(stock).loc[i-cur]['close']))

			if runs < 30: #If the date landed on a weekend, could try to look for days prior
				splitDate = date.split('/')
				date = splitDate[0] + '/' + str(int(splitDate[1]) - 1) + '/' + splitDate[2]

				splitDate = date.split('/')
				if int(splitDate[1]) < 1:
					date = str(int(splitDate[0]) - 1) + '/' + '28' + '/' + splitDate[2]

				splitDate = date.split('/')
				if int(splitDate[0]) < 1:
					date = '12' + '/' + '28' + '/' + str(int(splitDate[2])-1)

			else:
				found = True
				print('found empty')
				curPrice.append(None)



def main():
	fundamentals = pd.read_csv('fundamentals.csv')
	prices = pd.read_csv('prices-split-adjusted.csv')
	curPrice = []


	fundamentals = removeBadData(fundamentals)
	curPrice = findCurrentPrices(fundamentals, prices)


	fundamentals['Current Price'] = curPrice
	fundamentals.to_csv('fundamentalsWithPrice.csv', index = False)



if __name__ == "__main__":
	main()
