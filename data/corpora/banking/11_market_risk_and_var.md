# Market Risk, Value at Risk and Expected Shortfall

Market risk is the risk of losses from movements in market prices, such as interest rates, exchange rates, equity prices and commodity prices. It mainly affects a bank's trading book.

## Value at Risk

Value at Risk (VaR) estimates the loss that will not be exceeded with a given confidence level over a given horizon. A one-day 99% VaR of 10 million euros means that on 99 out of 100 days the portfolio is expected to lose less than 10 million euros.

There are three common ways to calculate VaR:

- **Historical simulation** revalues the current portfolio under price changes observed in the past, for example the last 250 or 500 trading days, and reads the loss at the chosen percentile.
- **Parametric (variance-covariance)** assumes returns follow a distribution, usually normal, and computes VaR from volatilities and correlations.
- **Monte Carlo simulation** generates many random scenarios from a statistical model and revalues the portfolio in each.

## Backtesting

Banks compare daily VaR with actual profit and loss. A day on which the loss exceeds VaR is an exception. Under the Basel traffic-light approach, too many exceptions over a year push a model into the yellow or red zone and increase its capital requirement, because the model is underestimating risk.

## Limitations of VaR

VaR says nothing about how large losses are once the threshold is breached, so two portfolios with the same VaR can have very different tail risk. VaR based on calm historical periods can also understate risk just before a crisis.

## Expected Shortfall

The Fundamental Review of the Trading Book (FRTB) replaces VaR with **Expected Shortfall (ES)** for internal models. Expected Shortfall is the average loss in the tail beyond the chosen percentile and is calibrated at 97.5%. Because it averages the tail, it captures the severity of extreme losses. FRTB also calibrates internal models to a period of significant market stress and uses different liquidity horizons for different risk factors.

## Stress Testing

Alongside statistical measures, banks run stress tests with hypothetical scenarios, such as a sudden rise in interest rates or a sharp fall in equity markets, to see losses that historical data may not contain.
