#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:06:50 2021

@author: nitinsinghal
"""

# RoboAdvisor Algo for multi asset portfolio
# Forecast future macro using past macro and FOMC/ECB/BoE forecasts
# Quantify peak/low of markets in relation to macro data MCP numerically
# Predict future asset price (1 month) using past macro data
# Calculate return of each asset for future timeframe (1m - 1 yr - 5yr - 10yr - 15 yr)
# Bonds calculate yield income (Actual/365) and price gain/loss. Price available for US, UK, EU. Don't know coupon rate
# Scenario analysis for different portfolio weights, for all portfolio returns, for time frame
# Account for correlation.
# Tax efficiency will be done later.
# Create backtesting logic for each portfolio return

#Import libraries
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Import the macro and asset data for each country
# Macro data - US, UK, EU, India, China
USMacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/USMacrodata.csv')
EUMacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/EUMacrodata.csv')
UKMacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/UKMacrodata.csv')
#IndMacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/IndiaMacrodata_.csv')
#ChinaMacrodata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/ChinaMacrodata.csv')
# Asset data
Nasdaqdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/Nasdaq.csv')
SP500data = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/SP500.csv')
Russell2000data = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/Russell2000.csv')
CACdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/CAC40.csv')
FTSEdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/FTSE100.csv')
Daxdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/Dax.csv')
# Sensexdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/Sensex.csv')
# Shanghaidata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/ShanghaiComposite.csv')
OilWTIdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/OilWTI.csv')
Golddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/Gold.csv')
# Wilshire US Real Estate Securities Price Index (Wilshire US RESI) (WILLRESIPR)
WilshireREdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/WilshireREPriceIndex.csv')
US10YrYielddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/US10YrYield.csv')
UK10YrYielddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/UK10YrYield.csv')
EUBund10YrYielddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/German10YrYield.csv')
#India10YrYielddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/India10YrYield.csv')
#China10YrYielddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/China10YrYield.csv')
UST10YrPriceddata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/UST10YrPrice.csv')
UKGilt10YrPricedata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/UKGilt10YrPrice.csv')
EUBund10YrPricedata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/EUBund10YrPrice.csv')
FedForecastdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/FedForecasts.csv')
EUForecastdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/ECBForecasts.csv')
BoEForecastdata = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/BoEForecasts.csv')
PortfolioWeights = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/PortfolioWeights.csv')

GBPUSD = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/GBPUSD.csv')
EURUSD = pd.read_csv('/Users/nitinsinghal/Dropbox/RoboAdvisorX/Data/EURUSD.csv')

# Format date coumn for each dataframe
def format_datetime(df):
    df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.sort_values(by=['Date']).reset_index(drop=True, inplace=True)
    return df
    
Nasdaqdata = format_datetime(Nasdaqdata)
SP500data = format_datetime(SP500data)
FTSEdata = format_datetime(FTSEdata)
Daxdata = format_datetime(Daxdata)
CACdata = format_datetime(CACdata)
# Sensexdata = format_datetime(Sensexdata)
# Shanghaidata = format_datetime(Shanghaidata)
Russell2000data = format_datetime(Russell2000data)
OilWTIdata = format_datetime(OilWTIdata)
Golddata = format_datetime(Golddata)
WilshireREdata = format_datetime(WilshireREdata)
UST10YrPriceddata = format_datetime(UST10YrPriceddata)
UKGilt10YrPricedata = format_datetime(UKGilt10YrPricedata)
EUBund10YrPricedata = format_datetime(EUBund10YrPricedata)
US10YrYielddata = format_datetime(US10YrYielddata)
UK10YrYielddata = format_datetime(UK10YrYielddata)
EUBund10YrYielddata = format_datetime(EUBund10YrYielddata)
#India10YrYielddata = format_datetime(India10YrYielddata)
#China10YrYielddata = format_datetime(China10YrYielddata)
GBPUSDdata = format_datetime(GBPUSD)
EURUSDdata = format_datetime(EURUSD)

# Take macro data from 1999 and formate date column
USMacro = USMacrodata[['Date','us_gdp_yoy', 'us_industrial_production','us_inflation_rate', 'us_core_pceinflation_rate',
                           'us_interest_rate','us_retail_sales_yoy', 'us_consumer_confidence', 'us_business_confidence',  
                           'us_unemployment_rate', 'us_manufacturing_production']] 
USMacro = format_datetime(USMacro)
USMacro = USMacro[(USMacro['Date'] > '1998-12-31')]

EUMacro = EUMacrodata[['Date','eu_gdp_yoy', 'eu_industrial_production','eu_inflation_rate', 'eu_core_inflation_rate',
                           'eu_interest_rate','eu_manufacturing_production','eu_retail_sales_yoy',
                           'eu_consumer_confidence','eu_business_confidence','eu_unemployment_rate']]
EUMacro = format_datetime(EUMacro)
EUMacro = EUMacro[(EUMacro['Date'] > '1998-12-31')]

UKMacro = UKMacrodata[['Date','uk_gdp_yoy', 'uk_industrial_production','uk_inflation_rate', 'uk_core_inflation_rate',
                           'uk_interest_rate','uk_manufacturing_production','uk_retail_sales_yoy',
                           'uk_consumer_confidence','uk_business_confidence','uk_unemployment_rate']]
UKMacro = format_datetime(UKMacro)
UKMacro = UKMacro[(UKMacro['Date'] > '1998-12-31')]

# IndMacro = IndMacrodata[['Date',	'in_consumer_confidence','in_gdp_yoy','in_industrial_production', 
#                              'in_interest_rate', 'in_manufacturing_production', 'in_wpi_inflation']]
# IndMacro = format_datetime(IndMacro)
# IndMacro = IndMacro[(IndMacro['Date'] > '1998-12-31')]

# ChinaMacro = ChinaMacrodata[['Date',	'ch_business_confidence','ch_consumer_confidence','ch_imports',
#                                  'ch_exports','ch_gdp_yoy','ch_industrial_production','ch_inflation_rate',
#                                  'ch_interest_rate','ch_retail_sales_yoy','ch_nonmanufacturing_pmi']]
# ChinaMacro = format_datetime(ChinaMacro)
# ChinaMacro = ChinaMacro[(ChinaMacro['Date'] > '1998-12-31')]

# merge us, eu, uk, india, china macro data files
Macro99 = pd.merge(USMacro, EUMacro, how='left', on='Date')
Macro99 = pd.merge(Macro99, UKMacro, how='left', on='Date')
#Macro99 = pd.merge(Macro99, IndMacro, how='left', on='Date')
#Macro99 = pd.merge(Macro99, ChinaMacro, how='left', on='Date')
Macro99.fillna(0, inplace=True)

# Use the 5 main macro factors (5MF) to predict asset prices for which cb's publish forecasts
# Merge the central bank forecasts into one dataframe
# Add the cb forecasts to the actual for each macro and predict asset price for each month

Macro5MF99act = Macro99[['Date','us_core_pceinflation_rate','us_inflation_rate','us_interest_rate','us_gdp_yoy','us_unemployment_rate', 
                     'eu_core_inflation_rate','eu_inflation_rate','eu_interest_rate','eu_gdp_yoy','eu_unemployment_rate',
                     'uk_core_inflation_rate','uk_inflation_rate','uk_interest_rate','uk_gdp_yoy','uk_unemployment_rate']]


# Use the high price from each asset dataframe. Remove ',', '-' characters
def get_price(asset):
    df = asset[['Date','High']]
    if(df['High'].dtype.kind == 'O'):
        df['High'] = df['High'].str.replace('-', '')
        df['High'] = pd.to_numeric(df['High'].str.replace(',', ''))
    return df

#Use High price instead of Close as it gives higher range to predict
Nasdaq = get_price(Nasdaqdata)
Nasdaq = Nasdaq.rename({'High':'Nasdaq'}, axis='columns')
SP500 = get_price(SP500data)
SP500 = SP500.rename({'High':'SP500'}, axis='columns')
Dax = get_price(Daxdata)
Dax = Dax.rename({'High':'Dax'}, axis='columns')
CAC = get_price(CACdata)
CAC = CAC.rename({'High':'CAC'}, axis='columns')
FTSE = get_price(FTSEdata)
FTSE = FTSE.rename({'High':'FTSE'}, axis='columns')
# Sensex = get_price(Sensexdata)
# Sensex = Sensex.rename({'High':'Sensex'}, axis='columns')
# Shanghai = get_price(Shanghaidata)
# Shanghai = Shanghai.rename({'High':'Shanghai'}, axis='columns')
Russell2000 = get_price(Russell2000data)
Russell2000 = Russell2000.rename({'High':'Russell2000'}, axis='columns')
OilWTI = get_price(OilWTIdata)
OilWTI = OilWTI.rename({'High':'OilWTI'}, axis='columns')
Gold = get_price(Golddata)
Gold = Gold.rename({'High':'Gold'}, axis='columns')
US10YrYield = get_price(US10YrYielddata)
US10YrYield = US10YrYield.rename({'High':'US10YrYield'}, axis='columns')
UK10YrYield = get_price(UK10YrYielddata)
UK10YrYield = UK10YrYield.rename({'High':'UK10YrYield'}, axis='columns')
EUBund10YrYield = get_price(EUBund10YrYielddata)
EUBund10YrYield = EUBund10YrYield.rename({'High':'EUBund10YrYield'}, axis='columns')
#India10YrYield = get_price(India10YrYielddata)
#India10YrYield = India10YrYield.rename({'High':'India10YrYield'}, axis='columns')
#China10YrYield = get_price(China10YrYielddata)
#China10YrYield = China10YrYield.rename({'High':'China10YrYield'}, axis='columns')
UST10YrPrice = get_price(UST10YrPriceddata)
UST10YrPrice = UST10YrPrice.rename({'High':'UST10YrPrice'}, axis='columns')
UKGilt10YrPrice = get_price(UKGilt10YrPricedata)
UKGilt10YrPrice = UKGilt10YrPrice.rename({'High':'UKGilt10YrPrice'}, axis='columns')
EUBund10YrPrice = get_price(EUBund10YrPricedata)
EUBund10YrPrice = EUBund10YrPrice.rename({'High':'EUBund10YrPrice'}, axis='columns')

WilshireREdata = WilshireREdata[~WilshireREdata['High'].str.startswith('.')]
WilshireREdata.reset_index(drop=True, inplace=True)
WilshireRE = get_price(WilshireREdata)
WilshireRE = WilshireRE.rename({'High':'WilshireRE'}, axis='columns')

#Merge stock, oil, Gold, RE index, Ccy data
MergedAssetdata = pd.merge(SP500, Nasdaq, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, Dax, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, CAC, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, FTSE, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, WilshireRE, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, Gold, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, OilWTI, how='left', on='Date')
#MergedAssetdata = pd.merge(MergedAssetdata, Russell2000, how='left', on='Date')
#MergedAssetdata = pd.merge(MergedAssetdata, Sensex, how='left', on='Date')
#MergedAssetdata = pd.merge(MergedAssetdata, Shanghai, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, GBPUSDdata, how='left', on='Date')
MergedAssetdata = pd.merge(MergedAssetdata, EURUSDdata, how='left', on='Date')
MergedAssetdata = MergedAssetdata.fillna(0)

#Get all merged data from 1999, as most indices have data from then 
MergedAsset99data = MergedAssetdata[(MergedAssetdata['Date'] > '1998-12-31')]
MergedAsset99data.drop_duplicates(subset='Date', keep='first', inplace=True)
MergedAsset99data = MergedAsset99data[~MergedAsset99data.eq(0).any(1)]
MergedAsset99data.reset_index(drop=True, inplace=True)
AllAsset99data = MergedAsset99data

# Merge Bond Price and Yield into one dataframe for countries
USBondYieldPrice = pd.merge(US10YrYield, UST10YrPrice, how='left', on='Date')
UKBondYieldPrice = pd.merge(UK10YrYield, UKGilt10YrPrice, how='left', on='Date')
EUBundYieldPrice = pd.merge(EUBund10YrYield, EUBund10YrPrice, how='left', on='Date')
AllBondYieldPricedata = pd.merge(USBondYieldPrice, UKBondYieldPrice, how='left', on='Date')
AllBondYieldPricedata = pd.merge(AllBondYieldPricedata, EUBundYieldPrice, how='left', on='Date')
AllBondYieldPricedata = AllBondYieldPricedata[(AllBondYieldPricedata['Date'] > '1998-12-31')]
AllBondYieldPricedata.drop_duplicates(subset='Date', keep='first', inplace=True)
AllBondYieldPricedata = AllBondYieldPricedata[~AllBondYieldPricedata.eq(0).any(1)]
AllBondYieldPricedata.reset_index(drop=True, inplace=True)
AllBondYieldPricedata.fillna(0, inplace=True)

# Merge the Central Bank 5 macro forecast data into one dataframe
# THIS DATA MUST BE UPDATED WITH ACTUAL EVERY MONTH AND RE-RUN THE ALGO
FedForecastdata = format_datetime(FedForecastdata)
EUForecastdata = format_datetime(EUForecastdata)
BoEForecastdata = format_datetime(BoEForecastdata)
CB5MFMergedactdata = pd.merge(FedForecastdata, EUForecastdata, on='Date', how='left')
CB5MFMergedactdata = pd.merge(CB5MFMergedactdata, BoEForecastdata, on='Date', how='left')

################################ START of METHODS ############################################

# Predict asset's monthly average price for next 3years using central banks 5MF macro 3 year forecast for US,UK,EU
def get_AssetMthlyPricePredUsingCBFrcst(allassetdata):
    colnames = list(allassetdata.columns)
    
    for i in range(1, len(colnames)):
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        # Resample asset price data monthly to calculate the mean
        assetdata.set_index(pd.DatetimeIndex(assetdata['Date']), inplace=True)
        assetdata.drop(['Date'],inplace=True, axis=1)
        assetmthlydata = assetdata.resample('M').mean()
        assetmthlydata['Date'] = assetmthlydata.index
        assetmthlydata['Date'] = pd.to_datetime(assetmthlydata['Date'].dt.strftime('%Y-%m'), format='%Y-%m')
        assetmthlydata.reset_index(drop=True, inplace=True)
        # Use actual macro data before first forecast data month
        CB5MFMergeddata = CB5MFMergedactdata
        Macro5MF99 = Macro5MF99act
        Macro5MF99 = Macro5MF99[(Macro5MF99['Date'] < CB5MFMergeddata.head(1)['Date'][0])]
        MergedAsset99Pred = pd.DataFrame()
        # Append one month CBForecast macro to 5MF macro data and predict asset price 
        # Loop till end of CBForecast data
        for j in range(0, len(CB5MFMergeddata)):
            # Append the frst CBForecast macro data row to all macrodata
            appnddate = CB5MFMergeddata.head(1)['Date'][j]
            Macro5MF99 = Macro5MF99.append(CB5MFMergeddata.head(1), ignore_index=True)
            # Select macro data upto t-1 for predicting asset price for t
            macrocbf99data = Macro5MF99[(Macro5MF99['Date'] < appnddate)]
            # Take asset price data upto t-1 for training the model
            assetmthlydata = assetmthlydata[(assetmthlydata['Date'] < appnddate)]
            assetmthlypricedata = assetmthlydata.drop(['Date'], axis=1)
            macromthlydata = macrocbf99data.drop(['Date'], axis=1)
            mergedassetmacro = pd.concat([macromthlydata, assetmthlypricedata], axis=1)
            # Create train test sets. Predict using Random Forest Algorithm
            X = mergedassetmacro.iloc[:, 0:len(mergedassetmacro.columns)-1].values
            y = mergedassetmacro.iloc[:,len(mergedassetmacro.columns)-1].values
            # X_train, X_test, y_train, y_test = train_test_split(Xasset, yasset, test_size = 0.25)
            rfreg = RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=2, max_depth=17,  
                                            min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
            rfreg.fit(X, y)
            # Predict next month asset value using latest trained model (t) and CB5MFrcsts macro data for (t+1) 
            X_pred = CB5MFMergeddata.head(1)
            X_pred.drop(['Date'],inplace=True, axis=1)
            rf_pred_nxtmth = rfreg.predict(X_pred)
            asset_predicted = pd.DataFrame([[pd.to_datetime(appnddate), rf_pred_nxtmth[0]]], columns=['Date',colnames[i]])
            MergedAsset99Pred = MergedAsset99Pred.append(asset_predicted, ignore_index=True)
            assetmthlydata = assetmthlydata.append(asset_predicted, ignore_index=True)
            # Drop the CBForecast row allready added to all macro data
            CB5MFMergeddata = CB5MFMergeddata.drop(j, axis=0)
        # Build the dataframe to return for all asset's predicted values
        if(i==1):
            AllAssetPreddata = MergedAsset99Pred
        else:
            AllAssetPreddata = pd.merge(AllAssetPreddata, MergedAsset99Pred, how='left', on='Date')
    return AllAssetPreddata

# Predict bond's monthly average price and yield for next 3years using central banks 5MF macro 3year forecast for US,UK,EU
def get_BondMthlyPriceYieldPredUsingCBFrcst(BondYieldPricedata):
    bondyieldprice = BondYieldPricedata
    colnames = list(BondYieldPricedata.columns)

    for i in range(1, len(colnames)):
        bonddata = pd.concat([bondyieldprice.iloc[:,0], bondyieldprice.iloc[:,i]], axis=1)
        # Resample bond price data monthly to calculate the mean
        bonddata.set_index(pd.DatetimeIndex(bonddata['Date']), inplace=True)
        bonddata.drop(['Date'],inplace=True, axis=1)
        bondmthlydata = bonddata.resample('M').mean()
        bondmthlydata['Date'] = bondmthlydata.index
        bondmthlydata['Date'] = pd.to_datetime(bondmthlydata['Date'].dt.strftime('%Y-%m'), format='%Y-%m')
        bondmthlydata.reset_index(drop=True, inplace=True)
        # Use actual macro data before first forecast data month
        CB5MFMergeddata = CB5MFMergedactdata
        Macro5MF99 = Macro5MF99act
        Macro5MF99 = Macro5MF99[(Macro5MF99['Date'] < CB5MFMergeddata.head(1)['Date'][0])]
        BondActPreddata = pd.DataFrame()
        # Append one month CBForecast macro to 5MF macro data and predict bond price 
        # Loop till end of CBForecast data
        for j in range(0, len(CB5MFMergeddata)):
            # Append the frst CBForecast macro data row to all macrodata
            appnddate = CB5MFMergeddata.head(1)['Date'][j]
            Macro5MF99 = Macro5MF99.append(CB5MFMergeddata.head(1), ignore_index=True)
            
            # Select macro data upto t-1 for predicting bond price for t
            macrocbf99data = Macro5MF99[(Macro5MF99['Date'] < appnddate)]
            
            # Take asset price data upto t-1 for training the model
            bondmthlydata = bondmthlydata[(bondmthlydata['Date'] < appnddate)]
            bondmthlytmin1data = bondmthlydata.drop(['Date'], axis=1)
            macromthlydata = macrocbf99data.drop(['Date'], axis=1)
            mergedbondmacro = pd.concat([macromthlydata, bondmthlytmin1data], axis=1)
            
            # Create train test sets. Predict using Random Forest Algorithm
            X = mergedbondmacro.iloc[:, 0:len(mergedbondmacro.columns)-1].values
            y = mergedbondmacro.iloc[:,len(mergedbondmacro.columns)-1].values
            # X_train, X_test, y_train, y_test = train_test_split(Xbond, ybond, test_size = 0.25)
            rfreg = RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=2, max_depth=17,  
                                            min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
            rfreg.fit(X, y)
            # Predict next month bond value using latest trained model (t) and CB5MFrcsts macro data for (t+1) 
            X_pred = CB5MFMergeddata.head(1)
            X_pred.drop(['Date'],inplace=True, axis=1)
            rf_pred_nxtmth = rfreg.predict(X_pred)
            bond_predicted = pd.DataFrame([[pd.to_datetime(appnddate), rf_pred_nxtmth[0]]], columns=['Date',colnames[i]])
            BondActPreddata = BondActPreddata.append(bond_predicted, ignore_index=True)
            bondmthlydata = bondmthlydata.append(bond_predicted, ignore_index=True)
            # Drop the CBForecast row allready added to all macro data
            CB5MFMergeddata = CB5MFMergeddata.drop(j, axis=0)
        # Build the dataframe to return for all bond's predicted values
        if(i==1):
            AllBondPreddata = BondActPreddata
        else:
            AllBondPreddata = pd.merge(AllBondPreddata, BondActPreddata, how='left', on='Date')
    AllBondPreddata.fillna(0, inplace=True)
    return AllBondPreddata

# Predict asset's monthly average price using actual Macro data (5MF/All) for US,UK,EU
# Use macro and asset data uptil t-2 to train. Then t-1 macro data to predict asset price for t
# Return predicted, actual, difference and accuracy data 
def get_AssetMthlyPricePredUsingMacroActual(allassetdata, allmacrodata):
    colnames = list(allassetdata.columns)
    
    for i in range(1, len(colnames)):
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        # Resample asset price data monthly to calculate the mean
        assetdata.set_index(pd.DatetimeIndex(assetdata['Date']), inplace=True)
        assetdata.drop(['Date'],inplace=True, axis=1)
        assetmthlydata = assetdata.resample('M').mean()
        assetmthlydata['Date'] = assetmthlydata.index
        assetmthlydata['Date'] = pd.to_datetime(assetmthlydata['Date'].dt.strftime('%Y-%m'), format='%Y-%m')
        assetmthlydata.reset_index(drop=True, inplace=True)
        # Use actual macro data 
        Macro5MFAct = allmacrodata
        AssetActPreddata = pd.DataFrame()
        # Use 5MF all macro & asset data upto t-2 to train and t-1 macro data to predict t asset price 
        # Loop in reverse, start at last month, drop 1 month each time
        # Till last 1 year (1999) of Macro data is left
        for j in range(len(allmacrodata)-1, 11, -1):
            # Drop last macro data row from allmacrodata
            traincutoffdate = Macro5MFAct.loc[j-2, 'Date']
            macropreddate = Macro5MFAct.loc[j-1, 'Date']
            assetpreddate = Macro5MFAct.loc[j, 'Date']
            
            # Select macro and asset data upto t-2 for training the model 
            macro5mftraindata = Macro5MFAct[(Macro5MFAct['Date'] <= traincutoffdate)]
            assetmthlytraindata = assetmthlydata[(assetmthlydata['Date'] <= traincutoffdate)]
            assetmthlypricedata = assetmthlytraindata.drop(['Date'], axis=1)
            macromthlydata = macro5mftraindata.drop(['Date'], axis=1)
            mergedassetmacro = pd.concat([macromthlydata, assetmthlypricedata], axis=1)
            
            # Create train test sets. Predict using Random Forest Algorithm
            X = mergedassetmacro.iloc[:, 0:len(mergedassetmacro.columns)-1].values
            y = mergedassetmacro.iloc[:,len(mergedassetmacro.columns)-1].values
            # X_train, X_test, y_train, y_test = train_test_split(Xasset, yasset, test_size = 0.25)
            rfreg = RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=2, max_depth=17,  
                                            min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
            rfreg.fit(X, y)
            
            # Take macro data for t-1 to predict asset price for t, using trained model (t-2)
            X_pred = Macro5MFAct[(Macro5MFAct['Date'] == macropreddate)]
            X_pred.drop(['Date'],inplace=True, axis=1)
            rf_pred_t = rfreg.predict(X_pred)
            asset_act_t = assetmthlydata[(assetmthlydata['Date'] == assetpreddate)]
    
            asset_pred_t = pd.DataFrame([[assetpreddate, rf_pred_t[0], asset_act_t.iloc[0,0]]], 
                                        columns=['Date',colnames[i]+'_Pred',colnames[i]+'_Act'])
            AssetActPreddata = AssetActPreddata.append(asset_pred_t, ignore_index=True)    
            # Drop the last (t-1) macrodata row used for prediction
            Macro5MFAct = Macro5MFAct.drop(j, axis=0)
        # Build the dataframe to return for all asset's predicted values
        if(i==1):
            AllAssetActPreddata = AssetActPreddata
        else:
            AllAssetActPreddata = pd.merge(AllAssetActPreddata, AssetActPreddata, how='left', on='Date')
    return AllAssetActPreddata



# Predict bond's monthly average price and yield using actual Macro (5MF/All)data for US,UK,EU
# Use macro and asset data uptil t-2 to train. Then t-1 macro data to predict asset price for t
# Return predicted, actual, difference and accuracy data 
def get_BondMthlyPriceYieldPredUsingMacroActual(bondyieldpricedata, allmacrodata):
    bondyieldprice = bondyieldpricedata
    colnames = list(bondyieldpricedata.columns)

    for i in range(1, len(colnames)):
        bonddata = pd.concat([bondyieldprice.iloc[:,0], bondyieldprice.iloc[:,i]], axis=1)
        # Resample bond price data monthly to calculate the mean
        bonddata = bonddata[~bonddata.eq(0).any(1)]
        bonddata.reset_index(drop=True, inplace=True)
        bonddata.set_index(pd.DatetimeIndex(bonddata['Date']), inplace=True)
        bonddata.drop(['Date'],inplace=True, axis=1)
        bondmthlydata = bonddata.resample('M').mean()
        bondmthlydata['Date'] = bondmthlydata.index
        bondmthlydata['Date'] = pd.to_datetime(bondmthlydata['Date'].dt.strftime('%Y-%m'), format='%Y-%m')
        bondmthlydata.reset_index(drop=True, inplace=True)
        # Use actual macro data
        allmacrovaliddata = allmacrodata[pd.to_datetime(allmacrodata['Date']) >= bondmthlydata.loc[0,'Date']]
        Macro99Act = allmacrovaliddata
        Macro99Act.reset_index(drop=True, inplace=True)
        BondActPreddata = pd.DataFrame()
        
        # Use all macro & bond data upto t-2 to train and t-1 macro data to predict t bond price 
        # Loop in reverse, start at last month, drop 1 month each time
        # Till last 1 year (1999 or mth/year for which bond price are not 0) of Macro data is left
        for j in range(len(allmacrovaliddata)-1, 11, -1):
            # Drop last macro data row from allmacrodata
            traincutoffdate = Macro99Act.loc[j-2, 'Date']
            macropreddate = Macro99Act.loc[j-1, 'Date']
            bondpreddate = Macro99Act.loc[j, 'Date']
            
            # Select macro and bond data upto t-2 for training the model 
            macro5mftraindata = Macro99Act[(Macro99Act['Date'] <= traincutoffdate)]
            bondmthlytraindata = bondmthlydata[(bondmthlydata['Date'] <= traincutoffdate)]
            bondmthlypriceylddata = bondmthlytraindata.drop(['Date'], axis=1)
            macromthlydata = macro5mftraindata.drop(['Date'], axis=1)
            mergedbondmacro = pd.concat([macromthlydata, bondmthlypriceylddata], axis=1)
            
            # Create train test sets. Predict using Random Forest Algorithm
            X = mergedbondmacro.iloc[:, 0:len(mergedbondmacro.columns)-1].values
            y = mergedbondmacro.iloc[:,len(mergedbondmacro.columns)-1].values
            # X_train, X_test, y_train, y_test = train_test_split(Xbond, ybond, test_size = 0.25)
            rfreg = RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=2, max_depth=17,  
                                            min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
            rfreg.fit(X, y)
            
            # Take macro data for t-1 to predict bond price for t, using trained model (t-2)
            X_pred = Macro99Act[(Macro99Act['Date'] == macropreddate)]
            X_pred.drop(['Date'],inplace=True, axis=1)
            rf_pred_t = rfreg.predict(X_pred)
            bond_act_t = bondmthlydata[(bondmthlydata['Date'] == bondpreddate)]
            
            bond_pred_t = pd.DataFrame([[bondpreddate, rf_pred_t[0], bond_act_t.iloc[0,0]]], 
                                        columns=['Date',colnames[i]+'_Pred',colnames[i]+'_Act'])
            BondActPreddata = BondActPreddata.append(bond_pred_t, ignore_index=True)
            
            # Drop the last (t-1) macrodata row used for prediction
            Macro99Act = Macro99Act.drop(j, axis=0)
        # Build the dataframe to return for all bond's predicted values
        if(i==1):
            AllBondActPreddata = BondActPreddata
        else:
            AllBondActPreddata = pd.merge(AllBondActPreddata, BondActPreddata, how='left', on='Date')
    AllBondActPreddata.fillna(0, inplace=True)
    return AllBondActPreddata

# Bond returns calculator. Bond data available only from Mar '08. Only for US, UK, EUBund
# BondNRR = [BondPrice(EoY-SoY)+BondYieldMean(EoY To SoY)]/BondPrice(SoY)

# Bond Annual NRR
def get_BondAnnualNRR(BondYieldPrice):
    bondyieldprice = BondYieldPrice
    cntrybondprice = bondyieldprice.filter(regex='Price|Date', axis=1)
    cntrybondyield = bondyieldprice.filter(regex='Yield|Date', axis=1)
        
    # Get list of countries for bond yield/price
    bondcolnames = list(bondyieldprice.columns)
    bondcntry = list()
    n=0
    for col in range(1, len(bondcolnames), 2):
        bondcntry.insert(n,bondcolnames[col][:2])
        n=n+1
    
    for i in range(1, len(bondcntry)+1):
        BondPrice = pd.concat([cntrybondprice.iloc[:,0], cntrybondprice.iloc[:,i]], axis=1)
        BondYield = pd.concat([cntrybondyield.iloc[:,0], cntrybondyield.iloc[:,i]], axis=1)
        colname = BondPrice.columns[1][:-5]
        
        BondPrice.drop_duplicates(subset='Date', keep='first', inplace=True)
        BondPrice = BondPrice[~BondPrice.eq(0).any(1)]
        BondPrice.reset_index(drop=True, inplace=True)
        BondPrice['Year'] = pd.to_datetime(BondPrice['Date']).dt.to_period('Y')
        BondPrice = BondPrice.drop(['Date'],axis=1)
        BondSoYdata = BondPrice.groupby('Year').first()
        BondSoYdata['Date'] = BondSoYdata.index
        BondSoYdata.rename(columns={BondSoYdata.columns[0]:colname+'_SoY'}, inplace=True)
        BondEoYdata = BondPrice.groupby('Year').last()
        BondEoYdata['Date'] = BondEoYdata.index
        BondEoYdata.rename(columns={BondEoYdata.columns[0]:colname+'_EoY'}, inplace=True)
        BondSoYEoYPricedata = pd.merge(BondSoYdata, BondEoYdata, how='left', on='Date')
        BondSoYEoYPricedata.reset_index(drop=True, inplace=True)
        
        BondYield.drop_duplicates(subset='Date', keep='first', inplace=True)
        BondYield = BondYield[~BondYield.eq(0).any(1)]
        BondYield.reset_index(drop=True, inplace=True)
        BondYield['Year'] = pd.to_datetime(BondYield['Date']).dt.to_period('Y')
        BondYield = BondYield.drop(['Date'],axis=1)
        BondYielddata = BondYield.groupby('Year').mean()
        BondYielddata['Date'] = BondYielddata.index
        BondYielddata.reset_index(drop=True, inplace=True)
        BondYielddata[colname] = BondYielddata[BondYield.columns[0]].div(100.00)
        BondSoYEoYPricedata = pd.merge(BondSoYEoYPricedata, BondYielddata, how='left', on='Date')
        
        BondSoYEoYPricedata[colname+'_NRR'] = (BondSoYEoYPricedata[colname+'_EoY'] - BondSoYEoYPricedata[colname+'_SoY']+
                           BondYielddata[colname])/BondSoYEoYPricedata[colname+'_SoY']
        # Build the dataframe to return for all bond's annual nrr values
        if(i==1):
            AllBondAnnlNRR = BondSoYEoYPricedata
        else:
            AllBondAnnlNRR = pd.merge(AllBondAnnlNRR, BondSoYEoYPricedata, how='left', on='Date')
    return AllBondAnnlNRR

# Bond Mthly NRR
def get_BondMthlyNRR(BondYieldPrice):
    bondyieldprice = BondYieldPrice
    cntrybondprice = bondyieldprice.filter(regex='Price|Date', axis=1)
    cntrybondyield = bondyieldprice.filter(regex='Yield|Date', axis=1)
        
    # Get list of countries for bond yield/price
    bondcolnames = list(bondyieldprice.columns)
    bondcntry = list()
    n=0
    for col in range(1, len(bondcolnames), 2):
        bondcntry.insert(n,bondcolnames[col][:2])
        n=n+1
    
    for i in range(1, len(bondcntry)+1):
        BondPrice = pd.concat([cntrybondprice.iloc[:,0], cntrybondprice.iloc[:,i]], axis=1)
        BondYield = pd.concat([cntrybondyield.iloc[:,0], cntrybondyield.iloc[:,i]], axis=1)
        colname = BondPrice.columns[1][:-5]
        
        BondPrice = BondPrice[~BondPrice.eq(0).any(1)]
        BondPrice.reset_index(drop=True, inplace=True)
        BondPrice['YrMth'] = pd.to_datetime(BondPrice['Date']).dt.to_period('M')
        BondPrice = BondPrice.drop(['Date'],axis=1)
        BondSoMdata = BondPrice.groupby('YrMth').first()
        BondSoMdata['Date'] = BondSoMdata.index
        BondSoMdata.rename(columns={BondSoMdata.columns[0]:colname+'_SoM'}, inplace=True)
        BondEoMdata = BondPrice.groupby('YrMth').last()
        BondEoMdata['Date'] = BondEoMdata.index
        BondEoMdata.rename(columns={BondEoMdata.columns[0]:colname+'_EoM'}, inplace=True)
        BondSoMEoMPricedata = pd.merge(BondSoMdata, BondEoMdata, how='left', on='Date')
        BondSoMEoMPricedata.reset_index(drop=True, inplace=True)
        
        BondYield.drop_duplicates(subset='Date', keep='first', inplace=True)
        BondYield = BondYield[~BondYield.eq(0).any(1)]
        BondYield.reset_index(drop=True, inplace=True)
        BondYield['YrMth'] = pd.to_datetime(BondYield['Date']).dt.to_period('M')
        BondYield = BondYield.drop(['Date'],axis=1)
        BondYielddata = BondYield.groupby('YrMth').mean()
        BondYielddata['Date'] = BondYielddata.index
        BondYielddata.reset_index(drop=True, inplace=True)
        BondYielddata[colname] = BondYielddata[BondYield.columns[0]].div(100.00)
        BondSoMEoMPricedata = pd.merge(BondSoMEoMPricedata, BondYielddata, how='left', on='Date')
        
        BondSoMEoMPricedata[colname+'_NRR'] = (BondSoMEoMPricedata[colname+'_EoM'] - BondSoMEoMPricedata[colname+'_SoM']+
                           BondYielddata[colname])/BondSoMEoMPricedata[colname+'_SoM']
        # Build the dataframe to return for all bond's monthly nrr values
        if(i==1):
            AllBondMthlyNRR = BondSoMEoMPricedata
        else:
            AllBondMthlyNRR = pd.merge(AllBondMthlyNRR, BondSoMEoMPricedata, how='left', on='Date')
    return AllBondMthlyNRR

# Bond Annualized Return data
# Calculate Annualized return using Monthly Net Return rate
def get_BondAnnlzdReturn(BondYieldPrice):
    bondyieldprice = BondYieldPrice
    cntrybondprice = bondyieldprice.filter(regex='Price|Date', axis=1)
    cntrybondyield = bondyieldprice.filter(regex='Yield|Date', axis=1)
        
    # Get list of countries for bond yield/price
    bondcolnames = list(bondyieldprice.columns)
    bondcntry = list()
    n=0
    for col in range(1, len(bondcolnames), 2):
        bondcntry.insert(n,bondcolnames[col][:2])
        n=n+1
    
    for i in range(1, len(bondcntry)+1):
        BondPrice = pd.concat([cntrybondprice.iloc[:,0], cntrybondprice.iloc[:,i]], axis=1)
        BondYield = pd.concat([cntrybondyield.iloc[:,0], cntrybondyield.iloc[:,i]], axis=1)
        colname = BondPrice.columns[1][:-5]
        
        BondPrice = BondPrice[~BondPrice.eq(0).any(1)]
        bondyear = pd.DatetimeIndex(BondPrice['Date']).year.unique().tolist()
        BondPrice['Year'] = pd.DatetimeIndex(BondPrice['Date']).year
        BondPrice['YrMth'] = pd.to_datetime(BondPrice['Date']).dt.to_period('M')
        BondPrice = BondPrice.drop(['Date'],axis=1)
        BondSoMdata = BondPrice.groupby('YrMth').first()
        BondSoMdata['Date'] = BondSoMdata.index
        BondSoMdata.rename(columns={BondSoMdata.columns[0]:colname+'_SoM', BondSoMdata.columns[1]:colname+'_Year'}, inplace=True)
        BondEoMdata = BondPrice.groupby('YrMth').last()
        BondEoMdata['Date'] = BondEoMdata.index
        BondEoMdata.rename(columns={BondEoMdata.columns[0]:colname+'_EoM'}, inplace=True)
        BondSoMEoMPricedata = pd.merge(BondSoMdata, BondEoMdata, how='left', on='Date')
        BondSoMEoMPricedata.reset_index(drop=True, inplace=True)
        
        BondYield.drop_duplicates(subset='Date', keep='first', inplace=True)
        BondYield = BondYield[~BondYield.eq(0).any(1)]
        BondYield.reset_index(drop=True, inplace=True)
        BondYield['YrMth'] = pd.to_datetime(BondYield['Date']).dt.to_period('M')
        BondYield = BondYield.drop(['Date'],axis=1)
        BondYielddata = BondYield.groupby('YrMth').mean()
        BondYielddata['Date'] = BondYielddata.index
        BondYielddata.reset_index(drop=True, inplace=True)
        BondYielddata[colname] = BondYielddata[BondYield.columns[0]].div(100.00)
        BondYieldPricedata = pd.merge(BondSoMEoMPricedata, BondYielddata, how='left', on='Date')
        
        BondYieldPricedata[colname+'_NRR'] = (BondYieldPricedata[colname+'_EoM'] - BondYieldPricedata[colname+'_SoM']+
                           BondYieldPricedata[colname])/BondYieldPricedata[colname+'_SoM']
        BondMthlyNRR = BondYieldPricedata[['Date', colname+'_NRR', colname+'_Year']]
        
        bondnrrdata = pd.concat([BondMthlyNRR.iloc[:,1], BondMthlyNRR.iloc[:,2]], axis=1)
        BondAnnlzdReturndata = pd.DataFrame(bondyear, columns=['Date'])
    
        for j in range(0,len(bondyear)):
            dfBond = bondnrrdata[(bondnrrdata[colname+'_Year'] == bondyear[j])]
            bondannlzdreturn = 1
            bondannlzdreturnval = 0
            for k in range(0,len(dfBond)):
                bondannlzdreturn = bondannlzdreturn * (1+dfBond.iloc[k,0])
            bondannlzdreturnval = pow(bondannlzdreturn,(1/len(dfBond))) - 1
            BondAnnlzdReturndata.loc[j,colname+'_AnnlzdRtrn'] = bondannlzdreturnval
        
        # Build the dataframe to return for all bond's monthly nrr values
        if(i==1):
            AllBondAnnlzdReturndata = BondAnnlzdReturndata
        else:
            AllBondAnnlzdReturndata = pd.merge(AllBondAnnlzdReturndata, BondAnnlzdReturndata, how='left', on='Date')
    return AllBondAnnlzdReturndata

# Calculate Annual Net Return rate using Start/End of Year value
# AR = (APEoY - APSoy)/APSoY
def get_AssetSoYEoYNRR(allassetdata):
    colnames = list(allassetdata.columns)
    for i in range(1, len(colnames)):
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        assetdata['Year'] = pd.to_datetime(assetdata['Date']).dt.to_period('Y')
        assetdata = assetdata.drop(['Date'],axis=1)
        AssetSoYdata = assetdata.groupby('Year').first()
        AssetSoYdata['Date'] = AssetSoYdata.index
        AssetSoYdata.rename(columns={AssetSoYdata.columns[0]:colnames[i]+'_SoY'}, inplace=True)
        AssetEoYdata = assetdata.groupby('Year').last()
        AssetEoYdata['Date'] = AssetEoYdata.index
        AssetEoYdata.rename(columns={AssetEoYdata.columns[0]:colnames[i]+'_EoY'}, inplace=True)
        AssetSoYEoYNRRdata = pd.merge(AssetSoYdata, AssetEoYdata, how='left', on='Date')
        AssetSoYEoYNRRdata.reset_index(drop=True, inplace=True)
        AssetSoYEoYNRRdata = AssetSoYEoYNRRdata.reindex(columns=['Date',colnames[i]+'_SoY',colnames[i]+'_EoY'])
        AssetSoYEoYNRRdata[colnames[i]+'_NRR'] = (AssetSoYEoYNRRdata[colnames[i]+'_EoY'] - 
                          AssetSoYEoYNRRdata[colnames[i]+'_SoY'])/AssetSoYEoYNRRdata[colnames[i]+'_SoY']
        if(i==1):
            AllAssetSoYEoYNRRdata = AssetSoYEoYNRRdata
        else:
            AllAssetSoYEoYNRRdata = pd.merge(AllAssetSoYEoYNRRdata, AssetSoYEoYNRRdata, how='left', on='Date')
    return AllAssetSoYEoYNRRdata

# Calculate Monthly Net Return rate using Start/End of Month value
# AR = (APEoM - APSoM)/APSoM
def get_AssetSoMEoMNRR(allassetdata):
    colnames = list(allassetdata.columns)
    for i in range(1, len(colnames)):
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        assetdata['YrMth'] = pd.to_datetime(assetdata['Date']).dt.to_period('M')
        assetdata = assetdata.drop(['Date'],axis=1)
        AssetSoMdata = assetdata.groupby('YrMth').first()
        AssetSoMdata['Date'] = AssetSoMdata.index
        AssetSoMdata.rename(columns={AssetSoMdata.columns[0]:colnames[i]+'_SoM'}, inplace=True)
        AssetEoMdata = assetdata.groupby('YrMth').last()
        AssetEoMdata['Date'] = AssetEoMdata.index
        AssetEoMdata.rename(columns={AssetEoMdata.columns[0]:colnames[i]+'_EoM'}, inplace=True)
        AssetSoMEoMNRRdata = pd.merge(AssetSoMdata, AssetEoMdata, how='left', on='Date')
        AssetSoMEoMNRRdata.reset_index(drop=True, inplace=True)
        AssetSoMEoMNRRdata = AssetSoMEoMNRRdata.reindex(columns=['Date',colnames[i]+'_SoM',colnames[i]+'_EoM'])
        AssetSoMEoMNRRdata[colnames[i]+'_NRR'] = (AssetSoMEoMNRRdata[colnames[i]+'_EoM'] - 
                          AssetSoMEoMNRRdata[colnames[i]+'_SoM'])/AssetSoMEoMNRRdata[colnames[i]+'_SoM']
        if(i==1):
            AllAssetSoMEoMNRRdata = AssetSoMEoMNRRdata
        else:
            AllAssetSoMEoMNRRdata = pd.merge(AllAssetSoMEoMNRRdata, AssetSoMEoMNRRdata, how='left', on='Date')
    return AllAssetSoMEoMNRRdata

# Calculate Annualized return using Monthly Net Return rate
# Asset Annualized Returns AAR = ((1+amr1)*(1+amr2)...*(1+amrn))^1/n - 1. amr = asset monthly return
def get_AssetAnnlzdReturn(allassetdata):
    colnames = list(allassetdata.columns)
    assetyear = pd.DatetimeIndex(allassetdata['Date']).year.unique().tolist()
    for i in range(1, len(colnames)):
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        assetdata['Year'] = pd.DatetimeIndex(assetdata['Date']).year
        assetdata['YrMth'] = pd.to_datetime(assetdata['Date']).dt.to_period('M')
        assetdata = assetdata.drop(['Date'],axis=1)
        AssetSoMdata = assetdata.groupby('YrMth').first()
        AssetSoMdata['Date'] = AssetSoMdata.index
        AssetSoMdata.rename(columns={AssetSoMdata.columns[0]:colnames[i]+'_SoM', AssetSoMdata.columns[1]:colnames[i]+'_Year'}, inplace=True)
        AssetEoMdata = assetdata.groupby('YrMth').last()
        AssetEoMdata['Date'] = AssetEoMdata.index
        AssetEoMdata.rename(columns={AssetEoMdata.columns[0]:colnames[i]+'_EoM'}, inplace=True)
        AssetSoMEoMNRRdata = pd.merge(AssetSoMdata, AssetEoMdata, how='left', on='Date')
        AssetSoMEoMNRRdata.reset_index(drop=True, inplace=True)
        AssetSoMEoMNRRdata = AssetSoMEoMNRRdata.reindex(columns=['Date',colnames[i]+'_SoM',colnames[i]+'_EoM',colnames[i]+'_Year'])
        AssetSoMEoMNRRdata[colnames[i]+'_NRR'] = (AssetSoMEoMNRRdata[colnames[i]+'_EoM'] - 
                          AssetSoMEoMNRRdata[colnames[i]+'_SoM'])/AssetSoMEoMNRRdata[colnames[i]+'_SoM']

        assetnrrdata = pd.concat([AssetSoMEoMNRRdata.iloc[:,3], AssetSoMEoMNRRdata.iloc[:,4]], axis=1)
        AssetAnnlzdReturndata = pd.DataFrame(assetyear, columns=['Date'])
    
        for j in range(0,len(assetyear)):
            dfAsset = assetnrrdata[(assetnrrdata[colnames[i]+'_Year'] == assetyear[j])]
            assetannlzdreturn = 1
            assetannlzdreturnval = 0
            for k in range(0,len(dfAsset)):
                assetannlzdreturn = assetannlzdreturn * (1+dfAsset.iloc[k,1])
            assetannlzdreturnval = pow(assetannlzdreturn,(1/len(dfAsset))) - 1
            AssetAnnlzdReturndata.loc[j,colnames[i]+'_AnnlzdRtrn'] = assetannlzdreturnval
        if(i==1):
            AllAssetAnnlzdReturndata = AssetAnnlzdReturndata
        else:
            AllAssetAnnlzdReturndata = pd.merge(AllAssetAnnlzdReturndata, AssetAnnlzdReturndata, how='left', on='Date')
    return AllAssetAnnlzdReturndata

# Average return is the average of monthly returns AvgR = Sum(amr)/n
def get_AssetAvgReturn(allassetdata):
    colnames = list(allassetdata.columns)
    assetyear = pd.DatetimeIndex(allassetdata['Date']).year.unique().tolist()
    for i in range(1, len(colnames)):
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        assetdata['Year'] = pd.DatetimeIndex(assetdata['Date']).year
        assetdata['YrMth'] = pd.to_datetime(assetdata['Date']).dt.to_period('M')
        assetdata = assetdata.drop(['Date'],axis=1)
        AssetSoMdata = assetdata.groupby('YrMth').first()
        AssetSoMdata['Date'] = AssetSoMdata.index
        AssetSoMdata.rename(columns={AssetSoMdata.columns[0]:colnames[i]+'_SoM', AssetSoMdata.columns[1]:colnames[i]+'_Year'}, inplace=True)
        AssetEoMdata = assetdata.groupby('YrMth').last()
        AssetEoMdata['Date'] = AssetEoMdata.index
        AssetEoMdata.rename(columns={AssetEoMdata.columns[0]:colnames[i]+'_EoM'}, inplace=True)
        AssetSoMEoMNRRdata = pd.merge(AssetSoMdata, AssetEoMdata, how='left', on='Date')
        AssetSoMEoMNRRdata.reset_index(drop=True, inplace=True)
        AssetSoMEoMNRRdata = AssetSoMEoMNRRdata.reindex(columns=['Date',colnames[i]+'_SoM',colnames[i]+'_EoM',colnames[i]+'_Year'])
        AssetSoMEoMNRRdata[colnames[i]+'_NRR'] = (AssetSoMEoMNRRdata[colnames[i]+'_EoM'] - 
                          AssetSoMEoMNRRdata[colnames[i]+'_SoM'])/AssetSoMEoMNRRdata[colnames[i]+'_SoM']

        assetnrrdata = pd.concat([AssetSoMEoMNRRdata.iloc[:,3], AssetSoMEoMNRRdata.iloc[:,4]], axis=1)
        AssetAvgReturndata = pd.DataFrame(assetyear, columns=['Date'])
    
        for j in range(0,len(assetyear)):
            dfAsset = assetnrrdata[(assetnrrdata[colnames[i]+'_Year'] == assetyear[j])]
            assetavgreturn = 0
            assetavgreturnval = 0
            for k in range(0,len(dfAsset)):
                assetavgreturn = assetavgreturn + dfAsset.iloc[k,1]
            assetavgreturnval = assetavgreturn/len(dfAsset)
            AssetAvgReturndata.loc[j,colnames[i]+'_AvgRtrn'] = assetavgreturnval
        if(i==1):
            AllAssetAvgReturndata = AssetAvgReturndata
        else:
            AllAssetAvgReturndata = pd.merge(AllAssetAvgReturndata, AssetAvgReturndata, how='left', on='Date')
    return AllAssetAvgReturndata

# Calculate Asset's Predicted Monthly Net Return rate using Curr Month - Prev month value
def get_AssetPredictedSoMEoMNRR(allassetdata):
    colnames = list(allassetdata.columns)
    for i in range(1, len(colnames)):
        AssetPredMthlyNRR = pd.DataFrame()
        assetdata = pd.concat([allassetdata.iloc[:,0], allassetdata.iloc[:,i]], axis=1)
        assetdata['YrMth'] = pd.to_datetime(assetdata['Date'], yearfirst=True).dt.strftime('%Y-%m')
        assetdata = assetdata.drop(['Date'],axis=1)
        for j in range(0,(len(assetdata)-1)):
            AssetPredMthlyNRR.loc[j,'Date'] = assetdata.iloc[j+1,1]
            AssetPredMthlyNRR.loc[j,colnames[i]+'_SoM'] = assetdata.iloc[j,0]
            AssetPredMthlyNRR.loc[j,colnames[i]+'_EoM'] = assetdata.iloc[j+1,0]
            AssetPredMthlyNRR.loc[j,colnames[i]+'_NRR'] = ((assetdata.iloc[j+1,0] - assetdata.iloc[j,0])/assetdata.iloc[j,0])
        if(i==1):
            AllAssetPredMthlyNRR = AssetPredMthlyNRR
        else:
            AllAssetPredMthlyNRR = pd.merge(AllAssetPredMthlyNRR, AssetPredMthlyNRR, how='left', on='Date')
    return AllAssetPredMthlyNRR

# Calculate Bond's Predicted Monthly Net Return rate using Curr Month - Prev month value
def get_BondPredictedMthlyNRR(BondYieldPrice):
    cntrybondprice = BondYieldPrice.filter(regex='Price|Date', axis=1)
    cntrybondyield = BondYieldPrice.filter(regex='Yield|Date', axis=1)
        
    # Get list of countries for bond yield/price
    bondcolnames = list(BondYieldPrice.columns)
    bondcntry = list()
    n=0
    for col in range(1, len(bondcolnames), 2):
        bondcntry.insert(n,bondcolnames[col][:2])
        n=n+1
    
    for i in range(1, len(bondcntry)+1):
        BondPrice = pd.concat([cntrybondprice.iloc[:,0], cntrybondprice.iloc[:,i]], axis=1)
        BondYield = pd.concat([cntrybondyield.iloc[:,0], cntrybondyield.iloc[:,i]], axis=1)
        colname = BondPrice.columns[1][:-5]
        
        BondPrice = BondPrice[~BondPrice.eq(0).any(1)]
        BondPrice.reset_index(drop=True, inplace=True)
        BondPrice['YrMth'] = pd.to_datetime(BondPrice['Date'], yearfirst=True).dt.strftime('%Y-%m')
        BondPrice = BondPrice.drop(['Date'],axis=1)
        BondPredMthlyNRR = pd.DataFrame()
        for j in range(0,(len(BondPrice)-1)):
            BondPredMthlyNRR.loc[j,'Date'] = BondPrice.iloc[j+1,1]
            BondPredMthlyNRR.loc[j,colname+'_SoM'] = BondPrice.iloc[j,0]
            BondPredMthlyNRR.loc[j,colname+'_EoM'] = BondPrice.iloc[j+1,0]
    
        BondYield.drop_duplicates(subset='Date', keep='first', inplace=True)
        BondYield = BondYield[~BondYield.eq(0).any(1)]
        BondYield.reset_index(drop=True, inplace=True)
        BondYield['YrMth'] = pd.to_datetime(BondYield['Date'], yearfirst=True).dt.strftime('%Y-%m')
        BondYield = BondYield.drop(['Date'],axis=1)
        BondYield = BondYield.drop(BondYield.head(1).index)
        BondYield = BondYield.rename({'YrMth':'Date'}, axis='columns')
        BondYield.reset_index(drop=True, inplace=True)
        BondYield[BondYield.columns[0]] = BondYield[BondYield.columns[0]].div(100.00)
        BondYield = BondYield.rename({BondYield.columns[0]:colname+'_Yield'}, axis='columns')
        BondPredMthlyNRR = pd.merge(BondPredMthlyNRR, BondYield, how='left', on='Date')
        BondPredMthlyNRR[colname+'_NRR'] = (BondPredMthlyNRR[colname+'_EoM'] - BondPredMthlyNRR[colname+'_SoM']+
                           BondPredMthlyNRR[colname+'_Yield'])/BondPredMthlyNRR[colname+'_SoM']
        # Build the dataframe to return for all bond's annual nrr values
        if(i==1):
            AllBondPredMthlyNRR = BondPredMthlyNRR
        else:
            AllBondPredMthlyNRR = pd.merge(AllBondPredMthlyNRR, BondPredMthlyNRR, how='left', on='Date')
    return AllBondPredMthlyNRR

# Get Portfolio Predicted Assets Return based on type (index, bond) and return period (yr, m)
def get_PortfolioPredictedAssetsReturn(allassetdata, returnperiod, startyear):
    # allassetdata, returnperiod, startyear = AllAssetBondsFrcstPred3Yrdata, returnperiod, frcstpredstartyear
    allbonddata = allassetdata.filter(regex='10Yr|Date', axis=1)
    assetportfdata = allassetdata.drop(allassetdata.filter(regex='10Yr', axis=1), axis=1)
    PortfolioPredictedAssetsNRR = pd.DataFrame()

    # Get predicted asset return based on type (index, bond) and return period (yr, m)
    if returnperiod in ('1yr','2yr','3yr'):
        assetannlreturndata = get_AssetSoYEoYNRR(assetportfdata)
        assetannlreturndata = assetannlreturndata.filter(regex='_NRR|Date', axis=1)
        bondannlreturndata = get_BondAnnualNRR(allbonddata)
        bondannlreturndata =  bondannlreturndata.filter(regex='_NRR|Date', axis=1)
        allassetnrr = pd.merge(assetannlreturndata, bondannlreturndata, how='left', on='Date')
        allassetnrr.fillna(0, inplace=True)
        
        dfast = pd.DataFrame()
        for k, v in allassetnrr['Date'].iteritems():
            dfast.loc[k, 'Date'] = v.to_timestamp().year
        allassetnrr['Date'] = dfast['Date']
        
        if(returnperiod == '1yr'):
            endyear = startyear
            PortfolioPredictedAssetsNRR = allassetnrr[(allassetnrr['Date'] == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
        elif(returnperiod == '2yr'):
            endyear = startyear+1
            PortfolioPredictedAssetsNRR = allassetnrr[(allassetnrr['Date'] >= startyear)]
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR['Date'] <= endyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
        elif(returnperiod == '3yr'):
            endyear = startyear+2
            PortfolioPredictedAssetsNRR = allassetnrr[(allassetnrr['Date'] >= startyear)]
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR['Date'] <= endyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
    elif returnperiod in ('3m','6m','9m'):
        assetmthlyreturndata = get_AssetPredictedSoMEoMNRR(assetportfdata)
        assetmthlyreturndata = assetmthlyreturndata.filter(regex='_NRR|Date', axis=1)
        bondmthlyreturndata = get_BondPredictedMthlyNRR(allbonddata)
        bondmthlyreturndata = bondmthlyreturndata.filter(regex='_NRR|Date', axis=1)
        allassetnrr = pd.merge(assetmthlyreturndata, bondmthlyreturndata, how='left', on='Date')
        allassetnrr.fillna(0, inplace=True)

        if(returnperiod == '3m'):
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR.index < 3)]
        elif(returnperiod == '6m'):
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR.index < 6)]
        elif(returnperiod == '9m'):
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR.index < 9)]
    else:
        print('Invalid return period')
    return PortfolioPredictedAssetsNRR

# Calculate Portfolio Predicted Return for different timeframes (1m-3yr)
def get_PortfolioPredictedReturn(allassetdata, returnperiod, startyear, cashreturn):
    # Call method to get Portfolio Assets Return based on type (index, bond) and return period (yr, m)
    portfolioassetsnrr = get_PortfolioPredictedAssetsReturn(allassetdata, returnperiod, startyear)
    # portfolioassetsnrr = get_PortfolioPredictedAssetsReturn(AllAssetBondsFrcstPred3Yrdata, returnperiod, frcstpredstartyear)
    
    # Calculate portfolio weighted asset nrr return
    portfolioweights = PortfolioWeights[(PortfolioWeights['Period'] == returnperiod)]
    portfolioweights.drop(['Period','Total'], axis=1, inplace=True)
    #portfolioweights.drop(['Period','Total','REIndex','Gold','OilWTI'], axis=1, inplace=True)
    portfolioweights.reset_index(drop=True, inplace=True)
    
    # Filter portfolio assets for one country
    Portfolio = portfolioassetsnrr.filter(regex='Date|Nasdaq|UST10Yr|WilshireRE|Gold|OilWTI', axis=1)
    #Portfolio = portfolioassetsnrr.filter(regex='Date|SP500|UKGilt', axis=1)
    Portfolio['Cash'] = np.array(cashreturn)
    portfoliodates = Portfolio['Date']
    Portfolio = Portfolio.drop(['Date'], axis=1)
    PortfolioPredictedReturn = pd.DataFrame()
    
    for i in range(0, len(portfoliodates)):
        PortfolioPredictedReturn.loc[i, 'Period'] = portfoliodates[i]
        PortfolioPredictedReturn.loc[i, 'Return'] = pd.np.multiply(Portfolio.head(1),portfolioweights).sum(axis=1)[i]
        Portfolio = Portfolio.drop(i, axis=0)
    return PortfolioPredictedReturn

# Calculate Portfolio Predicted Variance (Volatility) for different timeframes (1m-3yr)
def get_PortfolioPredictedVariance(allassetdata, returnperiod, startyear):
    allbonddata = allassetdata.filter(regex='10Yr|Date', axis=1)
    assetportfdata = allassetdata.drop(allassetdata.filter(regex='10Yr', axis=1), axis=1)
    PortfolioPredictedAssetsNRR = pd.DataFrame()
    
    # Get portfolio predicted assets monthly return based on type (index, bond)
    assetmthlyreturndata = get_AssetPredictedSoMEoMNRR(assetportfdata)
    assetmthlyreturndata = assetmthlyreturndata.filter(regex='_NRR|Date', axis=1)
    bondmthlyreturndata = get_BondPredictedMthlyNRR(allbonddata)
    bondmthlyreturndata = bondmthlyreturndata.filter(regex='_NRR|Date', axis=1)
    allassetnrr = pd.merge(assetmthlyreturndata, bondmthlyreturndata, how='left', on='Date')
    allassetnrr.fillna(0, inplace=True)
    
    # Get portfilio for period (yr, m)
    if returnperiod in ('1yr','2yr','3yr'):
        if(returnperiod == '1yr'):
            endyear = startyear
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
        elif(returnperiod == '2yr'):
            endyear = startyear+1
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year >= startyear)]
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(pd.DatetimeIndex(PortfolioPredictedAssetsNRR['Date']).year <= endyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
        elif(returnperiod == '3yr'):
            endyear = startyear+2
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year >= startyear)]
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(pd.DatetimeIndex(PortfolioPredictedAssetsNRR['Date']).year <= endyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
    elif returnperiod in ('3m','6m','9m'):
        if(returnperiod == '3m'):
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR.index < 3)]
        elif(returnperiod == '6m'):
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR.index < 6)]
        elif(returnperiod == '9m'):
            PortfolioPredictedAssetsNRR = allassetnrr[(pd.DatetimeIndex(allassetnrr['Date']).year == startyear)]
            PortfolioPredictedAssetsNRR.reset_index(drop=True, inplace=True)
            PortfolioPredictedAssetsNRR = PortfolioPredictedAssetsNRR[(PortfolioPredictedAssetsNRR.index < 9)]
    else:
        print('Invalid variance period')
    
    # Get portfolio weights
    portfolioweights = PortfolioWeights[(PortfolioWeights['Period'] == returnperiod)]
    portfolioweights.drop(['Period','Total'], axis=1, inplace=True)
    # portfolioweights.drop(['Period','Total','REIndex','Gold','OilWTI'], axis=1, inplace=True)
    portfolioweights.reset_index(drop=True, inplace=True)
    
    # Filter portfolio assets for one country
    Portfolio = PortfolioPredictedAssetsNRR.filter(regex='Date|Nasdaq|UST10Yr|WilshireRE|Gold|OilWTI', axis=1)
    #Portfolio = PortfolioPredictedAssetsNRR.filter(regex='Date|SP500|UKGilt', axis=1)
    # Calculate weighted variance for each assets nrr. 
    # Var x = ∑((x-Avgx)^2)/n. Weighted Var = (Vari*Wi^2)
    PredictedAssetsWghtVar = pd.DataFrame()
    for i in range(1, len(Portfolio.columns)):
        colname = Portfolio.columns[i][:-4]
        PredictedAssetsWghtVar.loc[0, colname+'_WghtVar'] = np.square(portfolioweights.iloc[0,i-1]) * Portfolio.iloc[:,i].var()
    
    # Calculate weighted covariance for assetnrr unique pairs. 
    # Cov(AR1,AR2) = CorrCoeffAR1AR2*SDAR1*SDAR2. weighted cov = (2*Wi*Wj*Cov(ARi,ARj)
    rowcount = 0
    for m in range(1, len(Portfolio.columns)-1):
        rowcount += m
    PredictedAssetsWghtCoVar = pd.DataFrame()
    n=0
    while (n<rowcount):
        for i in range(1, len(Portfolio.columns)):
            colname = Portfolio.columns[i][:-4]
            for j in range(i+1, len(Portfolio.columns)):
                AssetsCoVar = Portfolio[[Portfolio.columns[i], Portfolio.columns[j]]].cov().iloc[0,1]
                PredictedAssetsWghtCoVar.loc[n, 'WghtCoVar'] = 2 * portfolioweights.iloc[0,i-1] * portfolioweights.iloc[0,j-1] * AssetsCoVar
                n += 1
    
    # Calculate variance for portfolio 
    # Var(Portf) = ∑i=1 to n (VarARi*Wi^2) + ∑i=1 to n∑j=i+1 to n (2*Wi*Wj*Cov(ARi,ARj))
    PortfolioPredictedAssetsVar = PredictedAssetsWghtVar.sum(axis=1)[0] + PredictedAssetsWghtCoVar.sum(axis=0)[0]
    return PortfolioPredictedAssetsVar

# Direction - Calculate Predicted vs Actual direction of all actual asset/bond prices
def get_AssetBondActualPredictedDirection(acolnames, allassetpredicteddata, bcolnames, allbondtpredicteddata):
    AllAssetActPredDirection = pd.DataFrame()
    for i in range(1, len(acolnames)):
        AssetActPredDirection = pd.DataFrame()
        for m in range(0, len(allassetpredicteddata)-1):
            AssetActPredDirection.loc[m, 'Date'] = allassetpredicteddata.loc[m,'Date']
            AssetActPredDirection.loc[m, acolnames[i]+'_PredDirDiff'] = allassetpredicteddata.loc[m, acolnames[i]+'_Pred'] - allassetpredicteddata.loc[m+1, acolnames[i]+'_Pred']
            AssetActPredDirection.loc[m, acolnames[i]+'_ActDirDiff'] = allassetpredicteddata.loc[m, acolnames[i]+'_Act'] - allassetpredicteddata.loc[m+1, acolnames[i]+'_Act']
            if(np.sign(AssetActPredDirection.loc[m, acolnames[i]+'_PredDirDiff']) == np.sign(AssetActPredDirection.loc[m, acolnames[i]+'_ActDirDiff'])):
                AssetActPredDirection.loc[m, acolnames[i]+'_ActPredSameDir'] = 1
            else:
                AssetActPredDirection.loc[m, acolnames[i]+'_ActPredSameDir'] = 0
        if(i==1):
            AllAssetActPredDirection = AssetActPredDirection
        else:
            AllAssetActPredDirection = pd.merge(AllAssetActPredDirection, AssetActPredDirection, how='left', on='Date')
    
    AllBondActPredDirection = pd.DataFrame()
    for j in range(1, len(bcolnames)):
        BondActPredDirection = pd.DataFrame()
        for n in range(0, len(allbondtpredicteddata)-1):
            BondActPredDirection.loc[n, 'Date'] = allbondtpredicteddata.loc[n,'Date']
            BondActPredDirection.loc[n, bcolnames[j]+'_PredDirDiff'] = allbondtpredicteddata.loc[n, bcolnames[j]+'_Pred'] - allbondtpredicteddata.loc[n+1, bcolnames[j]+'_Pred']
            BondActPredDirection.loc[n, bcolnames[j]+'_ActDirDiff'] = allbondtpredicteddata.loc[n, bcolnames[j]+'_Act'] - allbondtpredicteddata.loc[n+1, bcolnames[j]+'_Act']
            if(np.sign( BondActPredDirection.loc[n, bcolnames[j]+'_PredDirDiff']) == np.sign(BondActPredDirection.loc[n, bcolnames[j]+'_ActDirDiff'])):
                BondActPredDirection.loc[n, bcolnames[j]+'_ActPredSameDir'] = 1
            else:
                BondActPredDirection.loc[n, bcolnames[j]+'_ActPredSameDir'] = 0
        if(j==1):
            AllBondActPredDirection = BondActPredDirection
        else:
            AllBondActPredDirection = pd.merge(AllBondActPredDirection, BondActPredDirection, how='left', on='Date')
     
    AllAssetBondActPredDirection = pd.merge(AllAssetActPredDirection, AllBondActPredDirection, how='left', on='Date')
    return AllAssetBondActPredDirection

# Calculate Actual data's predicted values difference and accuracy 
def get_AssetBondMthlyActualPredDiffAcrcy(allassetactualpreddata, allbondactualpreddata, acolnames, bcolnames):
    AllAssetActPredDiffAcrcydata = allassetactualpreddata
    for i in range(1, len(acolnames)):
        AllAssetActPredDiffAcrcydata[acolnames[i]+'_ActPredDiff'] = AllAssetActPredDiffAcrcydata[acolnames[i]+'_Act'] - AllAssetActPredDiffAcrcydata[acolnames[i]+'_Pred']
        AllAssetActPredDiffAcrcydata[acolnames[i]+'_ActPredAcrcy'] = (AllAssetActPredDiffAcrcydata[acolnames[i]+'_Act'] - AllAssetActPredDiffAcrcydata[acolnames[i]+'_Pred'])/AllAssetActPredDiffAcrcydata[acolnames[i]+'_Act']
    
    AllBondActPredDiffAcrcydata = allbondactualpreddata
    for j in range(1, len(bcolnames)):
        AllBondActPredDiffAcrcydata[bcolnames[j]+'_ActPredDiff'] = AllBondActPredDiffAcrcydata[bcolnames[j]+'_Act'] - AllBondActPredDiffAcrcydata[bcolnames[j]+'_Pred']
        AllBondActPredDiffAcrcydata[bcolnames[j]+'_ActPredAcrcy'] = (AllBondActPredDiffAcrcydata[bcolnames[j]+'_Act'] - AllBondActPredDiffAcrcydata[bcolnames[j]+'_Pred'])/AllBondActPredDiffAcrcydata[bcolnames[j]+'_Act']
    
    AllAssetBondActPredDiffAcrcydata = pd.merge(AllAssetActPredDiffAcrcydata, AllBondActPredDiffAcrcydata, how='left', on='Date')
    return AllAssetBondActPredDiffAcrcydata

# Calculate Forecast Predicted direction of 3yr forecasts of assets/bonds 
def get_AssetBondForecastPredictedDirection(acolnames, allassetsfrcstpred3yrdata, bcolnames, allbondsfrcstpred3yrdata):
    AllAssetFrcstPredDirection = pd.DataFrame()
    for i in range(1, len(acolnames)):
        AssetFrcstPredDirection = pd.DataFrame()
        for m in range(0, len(allassetsfrcstpred3yrdata)-1):
            AssetFrcstPredDirection.loc[m, 'Date'] = allassetsfrcstpred3yrdata.loc[m,'Date']
            AssetFrcstPredDirection.loc[m, acolnames[i]+'_FrcstPredDiff'] = allassetsfrcstpred3yrdata.loc[m, acolnames[i]] - allassetsfrcstpred3yrdata.loc[m+1, acolnames[i]]
            if(np.sign(AssetFrcstPredDirection.loc[m, acolnames[i]+'_FrcstPredDiff']) < 0):
                AssetFrcstPredDirection.loc[m, acolnames[i]+'_FrcstPredDir'] = -1
            else:
                AssetFrcstPredDirection.loc[m, acolnames[i]+'_FrcstPredDir'] = 1
        if(i==1):
            AllAssetFrcstPredDirection = AssetFrcstPredDirection
        else:
            AllAssetFrcstPredDirection = pd.merge(AllAssetFrcstPredDirection, AssetFrcstPredDirection, how='left', on='Date')
    
    AllBondFrcstPredDirection = pd.DataFrame()
    for j in range(1, len(bcolnames)):
        BondFrcstPredDirection = pd.DataFrame()
        for n in range(0, len(allbondsfrcstpred3yrdata)-1):
            BondFrcstPredDirection.loc[n, 'Date'] = allbondsfrcstpred3yrdata.loc[n,'Date']
            BondFrcstPredDirection.loc[n, bcolnames[j]+'_FrcstPredDiff'] = allbondsfrcstpred3yrdata.loc[n, bcolnames[j]] - allbondsfrcstpred3yrdata.loc[n+1, bcolnames[j]]
            if(np.sign(BondFrcstPredDirection.loc[n, bcolnames[j]+'_FrcstPredDiff']) <0 ):
                BondFrcstPredDirection.loc[n, bcolnames[j]+'_FrcstPredDir'] = -1
            else:
                BondFrcstPredDirection.loc[n, bcolnames[j]+'_FrcstPredDir'] = 1
        if(j==1):
            AllBondFrcstPredDirection = BondFrcstPredDirection
        else:
            AllBondFrcstPredDirection = pd.merge(AllBondFrcstPredDirection, BondFrcstPredDirection, how='left', on='Date')
     
    AllAssetBondFrcstPredDirection = pd.merge(AllAssetFrcstPredDirection, AllBondFrcstPredDirection, how='left', on='Date')
    return AllAssetBondFrcstPredDirection

################################ END of METHODS ############################################

# Calculate Annual NRR for all assets, including bonds
AllBondAnnlNRR = get_BondAnnualNRR(AllBondYieldPricedata)
AllBondAnnlNRR = AllBondAnnlNRR.filter(regex='_NRR|Date', axis=1)
AllBondAnnlNRR.fillna(0, inplace=True)
AllAssetSoYEoYNRRdata = get_AssetSoYEoYNRR(AllAsset99data)
AllAssetAnnlNRR = AllAssetSoYEoYNRRdata.filter(regex='_NRR|Date', axis=1)
AllAssetAnnlNRR = pd.merge(AllAssetAnnlNRR, AllBondAnnlNRR, how='left', on='Date')

# Calculate Monthly NRR for all assets, including bonds
AllBondMthlyNRR = get_BondMthlyNRR(AllBondYieldPricedata)
AllBondMthlyNRR = AllBondMthlyNRR.filter(regex='_NRR|Date', axis=1)
AllBondMthlyNRR.fillna(0, inplace=True)
AllAssetSoMEoMNRRdata = get_AssetSoMEoMNRR(AllAsset99data)
AllAssetMthlyNRR = AllAssetSoMEoMNRRdata.filter(regex='_NRR|Date', axis=1)
AllAssetMthlyNRR = pd.merge(AllAssetMthlyNRR, AllBondMthlyNRR, how='left', on='Date')

# Calculate Annualized NRR for all assets, including bonds
# Asset Annualized Returns = ((1+amr1)*(1+amr2)...*(1+amrn))^1/n - 1. amr = asset mthly return
AllBondAnnlzdReturndata = get_BondAnnlzdReturn(AllBondYieldPricedata)
AllBondAnnlzdReturndata.fillna(0, inplace=True)
AllAssetAnnlzdReturndata = get_AssetAnnlzdReturn(AllAsset99data)
AllAssetAnnlzdReturndata = pd.merge(AllAssetAnnlzdReturndata, AllBondAnnlzdReturndata, how='left', on='Date')
AllAssetAnnlzdReturndata.fillna(0, inplace=True)

# Set asset and bond data for testing
allassetdailydata = AllAsset99data #[['Date','SP500', 'Nasdaq']]
allbonddailydata = AllBondYieldPricedata #[['Date', 'UK10YrYield', 'UKGilt10YrPrice']]
allmacromthlydata =  Macro5MF99act #Macro99
acolnames = list(allassetdailydata.columns)
bcolnames = list(allbonddailydata.columns)

####### 3 year PREDICTION using Central Bank FORECAST #########
# Calculate predicted forecasts for next 3years, using CB 3yr forecast data for all assets/bonds
print('start 3yr asset pred calculation....', datetime.datetime.now())
AllAssetsFrcstPred3Yrdata = get_AssetMthlyPricePredUsingCBFrcst(allassetdailydata)
print('start 3yr bond pred calculation....', datetime.datetime.now())
AllBondFrcstPred3Yrdata = get_BondMthlyPriceYieldPredUsingCBFrcst(allbonddailydata)
print('merge 3yr asset bond preds....', datetime.datetime.now())
AllAssetBondsFrcstPred3Yrdata = pd.merge(AllAssetsFrcstPred3Yrdata, AllBondFrcstPred3Yrdata, how='left', on='Date')

# Get predicted forecasts difference, direction
AllAssetBondFrcstPredDirection = get_AssetBondForecastPredictedDirection(acolnames, AllAssetsFrcstPred3Yrdata, bcolnames, AllBondFrcstPred3Yrdata)
dffrcstpredtotaluppreddir = (AllAssetBondFrcstPredDirection.filter(regex='_FrcstPredDir', axis=1)== 1).sum(axis=0)
dffrcstpredtotaldownpreddir = (AllAssetBondFrcstPredDirection.filter(regex='_FrcstPredDir', axis=1)== -1).sum(axis=0)
pctfrcstpredtotaluppreddir = dffrcstpredtotaluppreddir.div(len(AllAssetBondFrcstPredDirection))
pctfrcstpredtotaldownpreddir = dffrcstpredtotaluppreddir.div(len(AllAssetBondFrcstPredDirection))
print(dffrcstpredtotaluppreddir, dffrcstpredtotaldownpreddir, pctfrcstpredtotaluppreddir, pctfrcstpredtotaldownpreddir)


###### Logic for BACKTESTING - actual data used for prediction #######
# Accuracy - Predicted - Actual prices using actual prices and macro data
# Direction - Calculate Predicted vs Actual direction using actual prices and macro data
# Calculate Predicted vs Actual portfolio return, volatility and risk
# Overall Portfolio Predicted vs Actual - Maximise (Return), Minimize (Vol, Risk)

# Get predicted vs actual values for actual data
allassetactpreddata = get_AssetMthlyPricePredUsingMacroActual(allassetdailydata, allmacromthlydata)
allbondactpreddata = get_BondMthlyPriceYieldPredUsingMacroActual(allbonddailydata, allmacromthlydata)
AllAssetBondActPreddata = pd.merge(allassetactpreddata, allbondactpreddata, how='left', on='Date')

# Get predicted vs actual difference and accuracy for actual data
AllAssetBondActPredDiffAcrcydata = get_AssetBondMthlyActualPredDiffAcrcy(allassetactpreddata, allbondactpreddata, acolnames, bcolnames)

# Get predicted vs actual direction for actual data
AllAssetBondActPredDirection = get_AssetBondActualPredictedDirection(acolnames, allassetactpreddata, bcolnames, allbondactpreddata)
crctdir = AllAssetBondActPredDirection.filter(regex='_ActPredSameDir', axis=1)
totcrctdir = crctdir.sum(axis=0)
totcrctdirpct = totcrctdir.div(len(AllAssetBondActPredDirection))
print(totcrctdir, len(AllAssetBondActPredDirection), totcrctdirpct)

# Calculate Total Annual ActPred Direction 
AllAssetSameDirYrly = AllAssetBondActPredDirection.filter(regex='Date|_ActPredSameDir', axis=1)
AllAssetSameDirYrly['Year'] = pd.to_datetime(AllAssetSameDirYrly['Date']).dt.to_period('Y')
AllAssetSameDirYrly = AllAssetSameDirYrly.drop(['Date'],axis=1)
AllAssetSameDirYrlyTotal = AllAssetSameDirYrly.groupby('Year').sum()
AllAssetSameDirYrlyTotal['Year'] = AllAssetSameDirYrlyTotal.index
AllAssetSameDirYrlyTotal.reset_index(drop=True, inplace=True)
# Plot the results, split by asset group type
assetcrctdiryr = AllAssetSameDirYrlyTotal.filter(regex='Date|SP500|Nasdaq|Dax|CAC|FTSE', axis=1)
cmdcrctdiryr = AllAssetSameDirYrlyTotal.filter(regex='Date|Wilshire|Gold|OilWTI|GBPUSD|EURUSD', axis=1)
bondcrctdiryr = AllAssetSameDirYrlyTotal.filter(regex='Date|US10YrYield|UST10YrPrice|UK10YrYield|UKGilt10YrPrice|EUBund10YrYield|EUBund10YrPrice', axis=1)
plt.figure()
assetcrctdiryr.plot.bar(subplots=True, figsize=(10,10), legend=False, title='ActVsPred Same Direction Mthly Total/Year')
plt.xlabel('index-Year. 0-2000, 19-2019')
plt.ylabel('Total out of 12')
plt.show()
cmdcrctdiryr.plot.bar(subplots=True, figsize=(10,10), legend=False, title='ActVsPred Same Direction Mthly Total/Year')
plt.xlabel('index-Year. 0-2000, 19-2019')
plt.ylabel('Total out of 12')
plt.show()
bondcrctdiryr.plot.bar(subplots=True, figsize=(10,10), legend=False, title='ActVsPred Same Direction Mthly Total/Year')
plt.xlabel('Yeindex-Yearar. 0-2000, 19-2019')
plt.ylabel('Total out of 12')
plt.show()

# Calculate Total Monthly ActPred Direction across years
AllAssetSameDirMthly = AllAssetBondActPredDirection.filter(regex='Date|_ActPredSameDir', axis=1)
AllAssetSameDirMthly['Month'] = pd.DatetimeIndex(AllAssetSameDirMthly['Date']).month
AllAssetSameDirMthlyTotal = AllAssetSameDirMthly.groupby('Month').sum()
# Calculate Percentage Monthly ActPred Direction across all years
AllAssetSameDirMthlyTotalJan = AllAssetSameDirMthlyTotal[AllAssetSameDirMthlyTotal.index == 1]/19
AllAssetSameDirMthlyTotalNotJan = AllAssetSameDirMthlyTotal[AllAssetSameDirMthlyTotal.index != 1]/20
AllAssetSameDirMthlyPct = AllAssetSameDirMthlyTotalJan.append(AllAssetSameDirMthlyTotalNotJan, ignore_index=True)
AllAssetSameDirMthlyPct['Month'] = AllAssetSameDirMthlyPct.index
AllAssetSameDirMthlyPct.reset_index(drop=True, inplace=True)
# Plot the results, split by asset group type
assetcrctdirmthpct = AllAssetSameDirMthlyPct.filter(regex='Date|SP500|Nasdaq|Dax|CAC|FTSE', axis=1)
cmdcrctdirmthpct = AllAssetSameDirMthlyPct.filter(regex='Date|Wilshire|Gold|OilWTI|GBPUSD|EURUSD', axis=1)
bondcrctdirmthpct = AllAssetSameDirMthlyPct.filter(regex='Date|US10YrYield|UST10YrPrice|UK10YrYield|UKGilt10YrPrice|EUBund10YrYield|EUBund10YrPrice', axis=1)
plt.figure()
assetcrctdirmthpct.plot.bar(subplots=True, figsize=(10,10), legend=False, title='ActVsPred Same Month Total for All Years %')
plt.xlabel('index-Month. 0-January, 11-December')
plt.ylabel('Total % out of 20 years')
plt.show()
cmdcrctdirmthpct.plot.bar(subplots=True, figsize=(10,10), legend=False, title='ActVsPred Same Month Total for All Years %')
plt.xlabel('index-Month. 0-January, 11-December')
plt.ylabel('Total % out of 20 years')
plt.show()
bondcrctdirmthpct.plot.bar(subplots=True, figsize=(10,10), legend=False, title='AActVsPred Same Month Total for All Years %')
plt.xlabel('index-Month. 0-January, 11-December')
plt.ylabel('Total % out of 20 years')
plt.show()

####### Logic for Portfolio Return  #######
# Portfolio Return = AR1*W1+Ar2*W2+...+ARn*Wn.
# Weight grid Example:  
    # Period 	Stk Ind	Bonds Gold	Rl Est Cash	Cmdt Indx	Tot
    # 3m	    20%	40%	20%	  10%	10%	   0%	0%          100%				
    # Asset = Stock Index, Bond, Gold, Oil, RE, Cash, Cmdt Indx.
# Calculate Annualized, Average return for a period
# Calculate Real (Inflation adjusted) and Nominal returns

# Calculate Forecast Predicted Portfolio Return for different timeframes (1m-3yr)
returnperiod = '3yr'
frcstpredstartyear = 2020
cashreturn = 0.0001
FrcstPredPortfolioReturn = get_PortfolioPredictedReturn(AllAssetBondsFrcstPred3Yrdata, returnperiod, frcstpredstartyear, cashreturn) 
print('\nFrcstPred Portfolio Weighted Return: ')
print('Period', 'Return')
for idx, row in FrcstPredPortfolioReturn.iterrows():
    print('%s   %.2f%%'%(row['Period'], row['Return']*100))

frcstpredportfannlzdreturn = 1.00
for i in range(0,len(FrcstPredPortfolioReturn)):
    frcstpredportfannlzdreturn = frcstpredportfannlzdreturn * (1+FrcstPredPortfolioReturn.loc[i,'Return'])
frcstpredportfannlzdreturnval = pow(frcstpredportfannlzdreturn,(1/len(FrcstPredPortfolioReturn))) - 1
print('\nFrcstPred Portfolio Annualized Return: %.2f%%'%(frcstpredportfannlzdreturnval*100))

frcstpredportfavgreturn = 0.00
for i in range(0,len(FrcstPredPortfolioReturn)):
    frcstpredportfavgreturn = frcstpredportfavgreturn + FrcstPredPortfolioReturn.loc[i,'Return']
frcstpredportfavgreturnval = frcstpredportfavgreturn/len(FrcstPredPortfolioReturn)
print('\nFrcstPredPortfolio Average Return: %.2f%%'%(frcstpredportfavgreturnval*100))

# Backtesting Actual - Calculate Actual Portfolio Return for different timeframes (1m-3yr-5yr-10yr)
returnperiod = '3yr'
actpredstartyear = 2017
cashreturn = 0.0001
allassetbondactualdata = AllAssetBondActPreddata.filter(regex='Date|_Act', axis=1)
ActPortfolioReturn = get_PortfolioPredictedReturn(allassetbondactualdata, returnperiod, actpredstartyear, cashreturn) 
print('\n Actual Portfolio Weighted Return: ')
print('Period', 'Return')
for idx, row in ActPortfolioReturn.iterrows():
    print('%s   %.2f%%'%(row['Period'], row['Return']*100))

actportfannlzdreturn = 1.00
for i in range(0,len(ActPortfolioReturn)):
    actportfannlzdreturn = actportfannlzdreturn * (1+ActPortfolioReturn.loc[i,'Return'])
actportfannlzdreturnval = pow(actportfannlzdreturn,(1/len(ActPortfolioReturn))) - 1
print('\n Actual Portfolio Annualized Return: %.2f%%'%(actportfannlzdreturnval*100))

actpredportfavgreturn = 0.00
for i in range(0,len(ActPortfolioReturn)):
    actpredportfavgreturn = actpredportfavgreturn + ActPortfolioReturn.loc[i,'Return']
actpredportfavgreturnval = actpredportfavgreturn/len(ActPortfolioReturn)
print('\n Actual Portfolio Average Return: %.2f%%'%(actpredportfavgreturnval*100))

# Backtesting Predicted - Calculate Predicted Portfolio Return for different timeframes (1m-3yr-5yr-10yr)
returnperiod = '3yr'
actpredstartyear = 2017
cashreturn = 0.0001
allassetbondactualpreddata = AllAssetBondActPreddata.filter(regex='Date|_Pred', axis=1)
PredPortfolioReturn = get_PortfolioPredictedReturn(allassetbondactualpreddata, returnperiod, actpredstartyear, cashreturn) 
print('\n Predicted Portfolio Weighted Return: ')
print('Period', 'Return')
for idx, row in PredPortfolioReturn.iterrows():
    print('%s   %.2f%%'%(row['Period'], row['Return']*100))

predportfannlzdreturn = 1.00
for i in range(0,len(PredPortfolioReturn)):
    predportfannlzdreturn = predportfannlzdreturn * (1+PredPortfolioReturn.loc[i,'Return'])
predportfannlzdreturnval = pow(predportfannlzdreturn,(1/len(PredPortfolioReturn))) - 1
print('\n Predicted Portfolio Annualized Return: %.2f%%'%(predportfannlzdreturnval*100))

predportfavgreturn = 0.00
for i in range(0,len(PredPortfolioReturn)):
    predportfavgreturn = predportfavgreturn + PredPortfolioReturn.loc[i,'Return']
predportfavgreturnval = predportfavgreturn/len(PredPortfolioReturn)
print('\n Predicted Portfolio Average Return: %.2f%%'%(predportfavgreturnval*100))


####### Logic for Portfolio Risk #######
# Var(Portf) = ∑i=1 to n (VarARi*Wi^2) + ∑i=1 to n∑j=i+1 to n (2*Wi*Wj*Cov(ARi,ARj)) = Volatility
# Cov(AR1, AR2) = CorrCoeffAR1AR2*SDAR1*SDAR2
# Var x = ∑((x-Avgx)^2)/n. Spread from mean. Excess weight to outliers
# Cov(x,y) = ∑[(xi-xavg)*(yi-yavg)]/n-1. Directional relationship between returns of 2 assets
# Correlation Coefficient - Strength of relationship between 2 assets -1 to 1
# Std Dev Portf = Sqrt(Var (Portf)). Dispersion of the returns from the mean. Same unit as data

# Calculate Forecast Predicted Portfolio Variance (Volatility) for different timeframes (1m-3yr)
FrcstPredPortfolioVariance = get_PortfolioPredictedVariance(AllAssetBondsFrcstPred3Yrdata, returnperiod, frcstpredstartyear)
FrcstPredPortfolioStdDev = np.sqrt(FrcstPredPortfolioVariance)
print('\n FrcstPred Portfolio Variance(Volatility): %.4f, Std Dev: %.4f' %(FrcstPredPortfolioVariance, FrcstPredPortfolioStdDev))

# Backtesting Actual - Calculate Actual Portfolio Variance (Volatility) for different timeframes (1m-3yr-5yr-10yr)
ActPortfolioVariance = get_PortfolioPredictedVariance(allassetbondactualdata, returnperiod, actpredstartyear)
ActPortfolioStdDev = np.sqrt(ActPortfolioVariance)
print('\n Actual Portfolio Variance(Volatility): %.4f, Std Dev: %.4f' %(ActPortfolioVariance, ActPortfolioStdDev))

# Backtesting Predicted - Calculate Predicted Portfolio Variance (Volatility) for different timeframes (1m-3yr-5yr-10yr)
PredPortfolioVariance = get_PortfolioPredictedVariance(allassetbondactualpreddata, returnperiod, actpredstartyear)
PredPortfolioStdDev = np.sqrt(PredPortfolioVariance)
print('\n Predicted Portfolio Variance(Volatility): %.4f, Std Dev: %.4f' %(PredPortfolioVariance, PredPortfolioStdDev))


# ####### Create the pipeline to run gridsearchcv for best estimator and hyperparameters ########
# def get_GridSearchHyperParams(X_train, X_test, y_train, y_test):
    
#     pipe_rf = Pipeline([('rgr', RandomForestRegressor(random_state=42))])
    
#     pipe_xgb = Pipeline([('rgr', XGBRegressor(objective ='reg:squarederror'))])
    
#     # Set grid search params
#     grid_params_rf = [{'rgr__n_estimators' : [1000],
#                        'rgr__criterion' : ['mse'], 
#                        'rgr__min_samples_leaf' : [2,3,4], 
#                        'rgr__max_depth' : [16,17,18],
#                        'rgr__min_samples_split' : [2,3,4],
#                        'rgr__max_features' : ['sqrt', 'log2']}]
    
#     grid_params_xgb = [{'rgr__learning_rate' : [0.1,0.2,0.3],
#                         'rgr__max_depth' : [3,4,5],
#                         'rgr__seed' : [1,2,3]}]
    
#     # Create grid search
#     gs_rf = GridSearchCV(estimator=pipe_rf,
#                          param_grid=grid_params_rf,
#                          scoring='neg_mean_squared_error',
#                          cv=10,
#                          n_jobs=-1)
    
#     gs_xgb = GridSearchCV(estimator=pipe_xgb,
#                           param_grid=grid_params_xgb,
#                           scoring='neg_mean_squared_error',
#                           cv=10,
#                           n_jobs=-1)
    
#     # List of grid pipelines
#     grids = [gs_rf, gs_xgb] 
#     # Grid dictionary for pipeline/estimator
#     grid_dict = {0:'RandomForestRegressor', 1: 'XGBoostRegressor'}
    
#     # Fit the pipeline of estimators using gridsearchcv
#     print('Fitting the gridsearchcv to pipeline of estimators...')
#     mse=0.0
#     rmse=0.0
#     mae=0.0
#     r2 = 0.0
#     resulterrorgrid = {}
    
#     for gsid,gs in enumerate(grids):
#         print('\nEstimator: %s. Start time: %s' %(grid_dict[gsid], datetime.datetime.now()))
#         gs.fit(X_train, y_train)
#         print('\n Best score : %.5f' % gs.best_score_)
#         print('\n Best grid params: %s' % gs.best_params_)
#         y_pred = gs.predict(X_test)
#         mse = mean_squared_error(y_test , y_pred)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_test , y_pred)
#         r2 = r2_score(y_test , y_pred)
#         resulterrorgrid[grid_dict[gsid]+'_best_params'] = gs.best_params_
#         resulterrorgrid[grid_dict[gsid]+'_best_score'] = gs.best_score_
#         resulterrorgrid[grid_dict[gsid]+'_mse'] = mse
#         resulterrorgrid[grid_dict[gsid]+'_rmse'] = rmse
#         resulterrorgrid[grid_dict[gsid]+'_mae'] = mae
#         resulterrorgrid[grid_dict[gsid]+'_r2'] = r2
#         testpreddiff = np.sort(y_pred-y_test, kind='quicksort')
#         testpreddiffpct = np.sort((y_pred-y_test)/y_pred, kind='quicksort')
#         resulterrorgrid[grid_dict[gsid]+'_testpreddiff'] = testpreddiff
#         resulterrorgrid[grid_dict[gsid]+'_testpreddiffpct'] = testpreddiffpct
#         print('\n Test set accuracy for best params MSE:%.4f, RMSE:%.4f, MAE:%.4f, R2:%.4f' 
#               %(mse, rmse, mae, r2))
    
#     return resulterrorgrid

# # Call GridSearchCV to see which is best algorithm
# bestalgo = {}
# bestalgo = get_GridSearchHyperParams(X_train, X_test, y_train, y_test)
# print(bestalgo)

# Test GirdSearchCV with one asset and one CB forecast

# allassetdailydata = AllAsset99data [['Date','SP500']]
# colnames = list(allassetdailydata.columns)
# assetdata = pd.concat([allassetdailydata.iloc[:,0], allassetdailydata.iloc[:,1]], axis=1)
# # Resample asset price data monthly to calculate the mean
# assetdata.set_index(pd.DatetimeIndex(assetdata['Date']), inplace=True)
# assetdata.drop(['Date'],inplace=True, axis=1)
# assetmthlydata = assetdata.resample('M').mean()
# assetmthlydata['Date'] = assetmthlydata.index
# assetmthlydata['Date'] = pd.to_datetime(assetmthlydata['Date'].dt.strftime('%Y-%m'), format='%Y-%m')
# assetmthlydata.reset_index(drop=True, inplace=True)
# # Use actual macro data before first forecast data month
# CB5MFMergeddata = CB5MFMergedactdata
# Macro5MF99 = Macro5MF99act
# Macro5MF99 = Macro5MF99[(Macro5MF99['Date'] < CB5MFMergeddata.head(1)['Date'][0])]
# MergedAsset99Pred = pd.DataFrame()

# # Append one month CBForecast macro to 5MF macro data and predict asset price 
# # Append the frst CBForecast macro data row to all macrodata
# appnddate = CB5MFMergeddata.head(1)['Date'][0]
# Macro5MF99 = Macro5MF99.append(CB5MFMergeddata.head(1), ignore_index=True)
# # Select macro data upto t-1 for predicting asset price for t
# macrocbf99data = Macro5MF99[(Macro5MF99['Date'] < appnddate)]
# # Take asset price data upto t-1 for training the model
# assetmthlydata = assetmthlydata[(assetmthlydata['Date'] < appnddate)]
# assetmthlypricedata = assetmthlydata.drop(['Date'], axis=1)
# macromthlydata = macrocbf99data.drop(['Date'], axis=1)
# mergedassetmacro = pd.concat([macromthlydata, assetmthlypricedata], axis=1)
# # Create train test sets. Predict using Random Forest Algorithm
# Xasset = mergedassetmacro.iloc[:, 0:len(mergedassetmacro.columns)-1].values
# yasset = mergedassetmacro.iloc[:,len(mergedassetmacro.columns)-1].values
# X_train, X_test, y_train, y_test = train_test_split(Xasset, yasset, test_size = 0.25)
# print(X_train, X_test, y_train, y_test)



# rfreg = RandomForestRegressor(n_estimators=200, criterion='mse', min_samples_leaf=2, max_depth=17,  
#                                 min_samples_split=2, max_features='sqrt', random_state=42, n_jobs=-1)
# rfreg.fit(X_train, y_train)
# # Predict next month asset value using latest trained model (t) and CB5MFrcsts macro data for (t+1) 
# X_pred = CB5MFMergeddata.head(1)
# X_pred.drop(['Date'],inplace=True, axis=1)
# rf_pred_nxtmth = rfreg.predict(X_pred)
# asset_predicted = pd.DataFrame([[pd.to_datetime(appnddate), rf_pred_nxtmth[0]]], columns=['Date',colnames[1]])
# MergedAsset99Pred = MergedAsset99Pred.append(asset_predicted, ignore_index=True)
# assetmthlydata = assetmthlydata.append(asset_predicted, ignore_index=True)
