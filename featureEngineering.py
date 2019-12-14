import argparse
import numpy as np
import pandas as pd
from random import random
from operator import add 
import statistics 
import datetime 

def removeIncomplete(fundamentals):
    # Remove incomplete entries - those with less than 4 annual financial statements
    # note - only removes 21 total entries
    for index, row in fundamentals.iterrows():
        if list(fundamentals['Ticker Symbol']).count(row['Ticker Symbol']) < 4:
            fundamentals = fundamentals.drop(index = index)

    fundamentals = fundamentals.drop(columns =['Unnamed: 0'])
    
    return fundamentals

def addIndustries(fundamentals, securities):
    sector = []
    
    for index, row in securities.iterrows():
        if row['Ticker symbol'] in list(fundamentals['Ticker Symbol']):
            for _ in range(4):
                sector.append(row['GICS Sector'])
                
    fundamentals['Industry'] = sector
    
    return fundamentals

def hotEncodeIndustries(xData):
    trainData = xData['Industry']
    trainDummy = pd.Series(trainData)
    a = pd.get_dummies(trainDummy)
    xData = xData.merge(a, left_index = True, right_index = True)
    xData = xData.drop(columns=['Industry'])
    
    return xData


def addYear(xData):
    years = xData['Period Ending']
    years = years.str[-2:]
    years = '20' + years
    xData['Year'] = years.astype(int)
    return xData

def liquidityRatios(xData):
    # provide information about a firm's liquidity (short term solvency)
    
    #current ratio, quick ratio, cash ratio - all present
    
    # simply need to add networking capital to total assets
    xData['NWC to Total Assets'] = (xData['Total Current Assets'] - xData['Total Current Liabilities'])/xData['Total Assets']
    
    return xData

def leverageRatios(xData):
    # indicate firm's ability to meet its long-term financial obligations
    
    # total debt ratio
    xData['Total Debt Ratio'] = (xData['Total Assets'] - xData['Total Equity'])/xData['Total Assets']
    
    # Debt-Equity Ratio
    xData['Debt-Equity Ratio'] = xData['Total Liabilities']/xData['Total Equity']
    
    # Equity Multiplier
    xData['Equity Multiplier'] = xData['Total Assets']/xData['Total Equity']
    
    # Long Term Debt Ratio
    xData['Long Term Debt Ratio'] = xData['Long-Term Debt']/(xData['Long-Term Debt']+xData['Total Equity'])
    
    # Times Interest Earned Ratio
    xData['Times Interest Earned Ratio'] = xData['Earnings Before Interest and Tax']/xData['Interest Expense']
    
    # Cash Coverage Ratio
    xData['Cash Coverage Ratio'] = (xData['Earnings Before Interest and Tax']+xData['Depreciation'])/xData['Interest Expense']
    
    return xData                                               
                                                             
def assetUtilizationRatios(xData):
    # How efficiently does a firm use its assets to generate sales
    
    # Inventory Turnover - how efficiently is inventory being managed
    xData['Inventory Turnover'] = (xData['Total Revenue']-xData['Gross Profit'])/xData['Inventory']
    
    # Days Sale in Inventory - how long does it take to sell off inventory
    xData['Days Sale in Inventory'] = 365 / xData['Inventory Turnover']
    
    # Receivables Turnover - how fast cash is actually collected on sales
    xData['Receivables Turnover'] = xData['Total Revenue']/xData['Accounts Receivable']
    
    # Days Sales in Receivables - the average collection period
    xData['Days Sales in Receivables'] = 365 / xData['Receivables Turnover']
    
    # Net Working Capital Turnover - how much work do we get from networking capital
    xData['NWC Turnover'] = xData['Total Revenue']/(xData['Total Current Assets'] - xData['Total Current Liabilities'])
    
    # Fixed Asset Turnover
    xData['Fixed Asset Turnover'] = xData['Total Revenue']/xData['Fixed Assets']
    
    # Total Asset Turnover
    xData['Total Asset Turnover'] = xData['Total Revenue']/xData['Total Assets']
    
    return xData


def profitabilityRatios(xData):
    # focus on profitability with emphasis on net income
    # profit margin - already present
    # return on equity - already present
    
    # return on assets - measure of profit per dollar of assets
    xData['ROA'] = xData['Net Income']/xData['Total Assets']
    
    return xData
                                                             

def marketValueRatios(xData):
    # understand how market values company
    
    # price to earnings ratio - measures how much investors are willing to pay per dollar of current earnings
    xData['PE Ratio'] = xData['Current Price']/(xData['Net Income']/xData['Estimated Shares Outstanding'])
    
    # price-sales ratio - more relevant ratio for companies lacking net income
    xData['Price-Sales Ratio'] = xData['Current Price']/(xData['Total Revenue']/xData['Estimated Shares Outstanding'])
    
    return xData
    
    
def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


def main():
    
    fundamentals = pd.read_csv('fundamentalsWithPrice.csv')
    securities = pd.read_csv('securities.csv')
    
    
    # remove incomplete
    fundamentals = removeIncomplete(fundamentals)
    
    # add industries
    fundamentals = addIndustries(fundamentals, securities)
    
    # hot encode industries
    fundamentals = hotEncodeIndustries(fundamentals)
    
    # conduct all feature engineering
    fundamentals = addYear(fundamentals)
    fundamentals = liquidityRatios(fundamentals)
    fundamentals = leverageRatios(fundamentals)
    fundamentals = assetUtilizationRatios(fundamentals)
    fundamentals = profitabilityRatios(fundamentals)
    fundamentals = marketValueRatios(fundamentals)
    
    
    # write CSV file
    fundamentals.to_csv('featureEngineeredData.csv', index = False)
    
if __name__ == "__main__":
    main()