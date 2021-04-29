RoboAdvisor Algo – Part 1

Objective
 - Overall - build a RoboAdvisor algo that can automatically construct and re-balance a multi-asset portfolio using forward predicted asset prices for different timeframes
 - Create a medium to long-term portfolio for small/medium investors (low investment amount > $500, covers majority of the population)
 - Part 1 - Predict 3 year forward monthly average asset prices and direction (stock indices, commodities, bonds, currencies) using historic macroeconomic and central bank forecasts for US, UK and EU. Only the 5 main macro factors were used
 - Part 1 - Predict forward Return and Risk for a multi-asset portfolio (stock indices, commodities, currencies, real estate, bonds) using pre-defined asset weights and predicted forward monthly asset prices for different time frames – 3 months to 3 years
 - Part 1 - Backtest the results by comparing predicted historic asset prices against actual for historic actual macro data
 - Part 2 – Find optimal portfolio weights for maximum return and minimum risk
 - Part 2 – Automate portfolio construction and re-balancing
 - Part 2 – Use India, China. Not considered due to lack of historic govt bond price data
 - Part 2 – Build a model to forecast macro data using all other macro data, to use for forward price prediction

Data
 - US, UK and EU monthly main macroeconomic data for last 21 years (Jan 99 – Dec 19) was used for independent variables (X).
 - Average monthly close price over the same period was used for asset prices (main stock indices, commodities, bonds, currencies) as the dependent variable (Y).
 - All 3 countries’ macroeconomic data was combined to see their impact on each asset price. Only the 5 main macro factors (forecasted by central banks) were used for 3 year predictions – interest rate, core & headline inflation, GDP, unemployment
 - Historic Asset price, Macroeconomic data has been sourced from public data sets and our website  https://datawisdomx.com,  which sources data from reliable well-known data providers.

Legal Disclaimer – This research is not an investment advisory or a sales pitch.

Note - This research is based on a very simple premise and small data set. This by itself is not sufficient for all possible variations to the relationship between macroeconomic data, countries, asset prices, timeframes, algorithms used and other factors like political & central bank data, etc. Users can test that on their own and use as they see fit.

Disclaimer
Please use this research keeping in mind the disclaimer - datawisdomx, disclaimer
Please get in touch if you see any errors or want to discuss this further - nitin singhal, datawisdomx
