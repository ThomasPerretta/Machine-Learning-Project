import pandas as pd
import numpy as np



#Creates the percent dataset for the model

def main():
	xTrain = pd.read_csv('xTrain.csv')
	xTest = pd.read_csv('xTest.csv')
	yTrain = pd.read_csv('trainBinary.csv')
	yTest = pd.read_csv('testBinary.csv')



	def test(index, original):
		row = xTest.iloc[index]
		arr = []
		#print(row['Ticker Symbol'] + ' ' + original['Ticker Symbol'])
		for i,j in zip(row[1:89], original[1:89]):
			if j == 0:
				arr.append(0)
			else:
				arr.append((i-j)/j)

		return arr


	a = []
	abc = []

	curTic = ['AAL']
	original = xTrain.iloc[0]
	counter = 0

	for index, row in xTrain.iterrows():
		if row['Ticker Symbol'] != curTic:
			curTic = row['Ticker Symbol']
			a.append([0] * 88)
			original = row
			#test(counter, original)
			abc.append(test(counter, original))
			counter+=1
		else:
			b = []
			for i,j in zip(row[1:89], original[1:89]):
				if j == 0:
					b.append(0)
				else:
					b.append((i-j)/j)

			a.append(b)


	xNewTrain = pd.DataFrame(a, index=xTrain.iloc[:,1:89].index, columns=xTrain.iloc[:,1:89].columns)
	xNewTest = pd.DataFrame(abc, index=xTest.iloc[:,1:89].index, columns=xTest.iloc[:,1:89].columns)


	
	z1 = []
	z2 = []
	xNewTrain['Ticker Symbol'] = xTrain['Ticker Symbol']
	print(xNewTrain)
	curTic = ['ZZZ']
	for index, row in xNewTrain.iterrows():
		if row['Ticker Symbol'] != curTic:
			curTic = row['Ticker Symbol']
		else:
			z1.append(row.values)
			z2.append(yTrain.iloc[index].values[0])

	print(len(z1))
	print(z2)

	newTrain = pd.DataFrame(z1, columns = xNewTrain.columns)
	trainLabels = pd.DataFrame(z2, columns = yTrain.columns)

	

	newTrain.to_csv('xPercentTrain.csv', index = False)
	trainLabels.to_csv('percentTrainLabels.csv', index = False)


if __name__ == "__main__":
	main()