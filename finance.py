import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

ticker_symbols = [
    "HDFCBANK.NS",    # HDFC Bank
    "RELIANCE.NS",    # Reliance Industries
    "TCS.NS",         # Tata Consultancy Services
    "INFY.NS"         # Infosys
]

def calculate_returns(df):
    return ((df['Close'] - df['Open']) / df['Open']) * 100


returns = []
stock_names = []
all_data = {}

for ticker in ticker_symbols:
    print(f"Fetching data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start="2024-01-01", end="2024-12-31")
        
        if not df.empty:
            stock_names.append(ticker)
            returns.append(calculate_returns(df))
            all_data[ticker] = df
            print(f"✓ {ticker}: {len(df)} days of data fetched")
        else:
            print(f"✗ {ticker}: No data available")
    except Exception as e:
        print(f"✗ {ticker}: Error - {e}")


returns_df = pd.concat(returns, axis=1)
returns_df.columns = stock_names
returns_df.dropna(inplace=True)

mean_returns = returns_df.mean().values
cov_matrix = returns_df.cov().values
n = len(stock_names)
U = np.ones(n)

inv_cov = np.linalg.inv(cov_matrix)
w_mvp = np.dot(inv_cov, U) / np.dot(U.T, np.dot(inv_cov, U))

mvp_return = np.dot(w_mvp, mean_returns)
mvp_risk = np.sqrt(np.dot(w_mvp.T, np.dot(cov_matrix, w_mvp)))

print("\n✅ Minimum Variance Portfolio Results")
print("Stocks:", stock_names)
print("Weights (%):", np.round(w_mvp * 100, 2))
print("Expected Return:", round(mvp_return, 4))
print("Expected Risk (Std Dev):", round(mvp_risk, 4))


num_portfolios = 5000
random_weights = np.random.dirichlet(np.ones(n), num_portfolios)
random_returns = random_weights.dot(mean_returns)
random_risks = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))

plt.figure(figsize=(10, 6))
plt.scatter(random_risks, random_returns, c='lightgray', s=10, label='Random Portfolios')
plt.scatter(mvp_risk, mvp_return, c='red', marker='*', s=200, label='Minimum Variance Portfolio')
plt.plot([0, mvp_risk], [0.05, mvp_return], c='blue', linestyle='--', label='Capital Market Line')

plt.title(" MVP & Capital Market Line (CML)", fontsize=14)
plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Expected Return (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
