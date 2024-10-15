import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import os

##THIS IS WHAT YOU CHANGE
quantities = {
    'SNOW': 45.00,
    'AAPL': 3.057,
    'MSFT': 7.726,
    'AMZN': 40.00,
    'SOXQ': 267.344,
    'VOO': 11.6277,
    'VFIAX': 44.445
}


def get_market_data(tickers, period='5y', interval='1mo'):
    data = yf.download(tickers, period=period, interval=interval)
    return data['Adj Close']


def calculate_monthly_returns(prices):
    returns = prices.pct_change().dropna()
    return returns


def portfolio_returns(returns, weights):
    """
    Calculate the expected return of the portfolio.
    """
    # Multiply the returns by the weights and sum across assets
    weighted_returns = returns.mul(weights, axis=1).sum(axis=1)
    # Calculate the average return
    expected_return = weighted_returns.mean()
    return expected_return


def portfolio_stddev(returns, weights):
    """
    Calculate the standard deviation of the portfolio.
    """
    cov_matrix = returns.cov()
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std


# Fetch data
tickers = list(quantities.keys())
portfolio = get_market_data(tickers)
monthly_returns = calculate_monthly_returns(portfolio)
monthly_returns = monthly_returns[tickers]
market_returns = calculate_monthly_returns(get_market_data('SPY'))

# Get the annual risk-free rate in decimal form
risk_free = get_market_data('^TNX').iloc[-1] / 100  # Annual rate in decimal

# Calculate the total cash for each asset
cash = {
    ticker: quantities[ticker] * portfolio[ticker].iloc[-1]
    for ticker in tickers
}
total_cash = sum(cash.values())
#Weighting of assets
weights = {ticker: cash[ticker] / total_cash for ticker in tickers}
#Used for just array of weights
weights_array = np.array([weights[ticker] for ticker in tickers])

print("Current portfolio weights:")
for ticker, weight in zip(tickers, weights_array):
    print(f"{ticker}: {weight:.4f}")


def capm(returns, weights):
    beta_returns_df = calculate_monthly_returns(get_market_data('SPY'))
    beta_returns = beta_returns_df.values.flatten()
    weighted_returns = returns.mul(weights, axis=1).sum(axis=1)
    # Ensure the length of beta_returns and weighted_returns are matched
    min_length = min(len(beta_returns), len(weighted_returns))
    beta_returns = beta_returns[:min_length]
    weighted_returns = weighted_returns[:min_length]
    cov_matrix = np.cov(beta_returns, weighted_returns.values)
    # Extract the covariance between beta_returns and weighted_returns
    covar = cov_matrix[0, 1]
    # Compute the variance of the market returns
    market_var = cov_matrix[0, 0]
    beta = covar / market_var
    market_return = beta_returns.mean()
    market_return_annual = (1 + market_return)**12 - 1
    expected_return = risk_free + (beta * (market_return_annual - risk_free))
    return expected_return


def capm_standard_deviation(weights):
    """
    Calculates the portfolio standard deviation using CAPM-implied covariances.
    """
    tickers = ['SNOW', 'AAPL', 'MSFT', 'AMZN', 'SOXQ', 'VOO', 'VFIAX']
    min_length = min(len(monthly_returns), len(market_returns))

    # Slice returns to the same length
    sliced_market_returns = market_returns.values.flatten()[:min_length]
    sigma_m2 = np.var(sliced_market_returns)

    betas_list = []
    for ticker in tickers:
        # Get the returns for the ticker
        asset_returns = monthly_returns[ticker].values[:min_length]
        # Compute covariance
        covar = np.cov(asset_returns, sliced_market_returns)[0, 1]
        market_var = np.var(sliced_market_returns)
        beta = covar / market_var
        betas_list.append(beta)

    betas_array = np.array(betas_list)
    # Compute the covariance matrix using the outer product of betas
    cov_matrix = np.outer(betas_array, betas_array) * sigma_m2

    weights_array = np.array([weights[ticker] for ticker in tickers])
    portfolio_variance = np.dot(weights_array, np.dot(cov_matrix,
                                                      weights_array))
    portfolio_std = np.sqrt(portfolio_variance)

    return portfolio_std


capm_expected_return = capm(monthly_returns, weights_array)
print(f"\nCAPM Expected return of the portfolio: {capm_expected_return:.4f}")

capm_portfolio_std = capm_standard_deviation(weights)
print(f"CAPM Std Dev: {capm_portfolio_std:.4f}")

current_portfolio_returns = portfolio_returns(monthly_returns, weights)
current_portfolio_stddev = portfolio_stddev(monthly_returns, weights_array)
# Annualize current portfolio return and standard deviation
current_portfolio_return_annual = (1 + current_portfolio_returns)**12 - 1
current_portfolio_stddev_annual = current_portfolio_stddev * np.sqrt(12)
print(
    f"Current Portfolio Arithmetic Return: {current_portfolio_return_annual:.4f}"
)
print(f"Alpha: {current_portfolio_return_annual - capm_expected_return:.4f}")
print(f"Current Volatility (Std Dev): {current_portfolio_stddev_annual:.4f}")
current_sharpe_ratio = (current_portfolio_return_annual -
                        risk_free) / current_portfolio_stddev_annual
capm_sharpe = (capm_expected_return - risk_free) / capm_portfolio_std
print(f"Current Sharpe Ratio: {current_sharpe_ratio:.4f}")
print(f"Current CAPM Sharpe Ratio: {capm_sharpe:.4f}")

# Convert weights to array in the same order as tickers
initial_weights = np.array([weights[ticker] for ticker in tickers])

# Constraints: Sum of weights equals 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds: Each weight between 0 and 1
bounds = tuple((0, 1) for _ in range(len(initial_weights)))


def negative_sharpe_ratio(weights, returns, risk_free_rate):
    """
    Calculate the negative Sharpe ratio for optimization.
    """
    portfolio_return = portfolio_returns(returns, weights)
    portfolio_std = portfolio_stddev(returns, weights)
    # Annualize return and standard deviation
    portfolio_return_annual = (1 + portfolio_return)**12 - 1
    portfolio_std_annual = portfolio_std * np.sqrt(12)
    sharpe_ratio = (portfolio_return_annual -
                    risk_free_rate) / portfolio_std_annual
    return -sharpe_ratio


# Perform optimization
optimized = minimize(negative_sharpe_ratio,
                     initial_weights,
                     args=(monthly_returns, risk_free),
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)

# Extract optimal weights
optimal_weights = optimized.x

# Check if optimization was successful
if not optimized.success:
    print("Optimization failed:", optimized.message)
else:
    # Calculate optimal portfolio performance
    portfolio_return = portfolio_returns(monthly_returns, optimal_weights)
    portfolio_std = portfolio_stddev(monthly_returns, optimal_weights)
    # Annualize return and standard deviation
    portfolio_return_annual = (1 + portfolio_return)**12 - 1
    portfolio_std_annual = portfolio_std * np.sqrt(12)
    sharpe_ratio = (portfolio_return_annual - risk_free) / portfolio_std_annual

    # Print the results
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")

    print(f"\nExpected Annual Return: {portfolio_return_annual:.4f}")
    print(f"Annual Volatility (Std Dev): {portfolio_std_annual:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")