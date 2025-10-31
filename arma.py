
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats
import yfinance as yf

ticker_symbol = "HDFCBANK.NS" 
hdfc = yf.Ticker(ticker_symbol)
df = hdfc.history(start="2024-01-01", end="2024-12-31")
possible_cols = ['Close', 'Open']
col = next((c for c in possible_cols if c in df.columns), df.columns[-1])
prices = df[col].dropna().values

# Convert to log returns
y = np.diff(np.log(prices))
y = y - np.mean(y)

print(f"Loaded {len(y)} log returns from {col}")
print(f"Mean ≈ {np.mean(y):.6f}, Std ≈ {np.std(y):.6f}")


def acf(data, lag):
    n = len(data)
    c_k = []
    Y = np.mean(data)
    for k in range(lag + 1):
        sum_product = 0
        for t in range(n - k):
            p1 = data[t] - Y
            p2 = data[t + k] - Y
            sum_product += p1 * p2
        c_k.append(sum_product / n)
    P_k = np.array(c_k) / c_k[0] if c_k[0] != 0 else np.zeros(len(c_k))
    return P_k

def calculate_pacf(y, lags):
    rho = acf(y, lags)
    pacf_vals = [1.0]
    for k in range(1, lags + 1):
        P_k = np.array([[rho[abs(i - j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k + 1])
        phi_k = np.linalg.solve(P_k, rho_k)
        pacf_vals.append(phi_k[-1])
    return np.array(pacf_vals)


lag = 20
rho = acf(y, lag)
pac = calculate_pacf(y, lag)
band = 1.96 / np.sqrt(len(y))

plt.figure(figsize=(10,4))
plt.subplot(121); plt.stem(rho); plt.title("ACF"); plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.subplot(122); plt.stem(pac); plt.title("PACF"); plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.tight_layout(); plt.show()

def cutoff(series):
    vals = np.abs(series[1:])
    above = vals > band
    for i in range(1, len(above)):
        if not above[i] and not above[i-1]:
            return i
    return 1

p = cutoff(pac)
q = cutoff(rho)
print(f"Suggested p={p}, q={q}")


def arma_model(params, y, p, q):
    c = params[0]
    phi = params[1:p + 1]
    theta = params[p + 1:p + 1 + q]
    n = len(y)
    eps = np.zeros(n)
    for t in range(max(p, q), n):
        ar_term = np.dot(phi, y[t - p:t][::-1]) if p > 0 else 0
        ma_term = np.dot(theta, eps[t - q:t][::-1]) if q > 0 else 0
        eps[t] = y[t] - (c + ar_term + ma_term)
    return eps

def fit_arma_manual(y, p, q):
    init_params = np.zeros(1 + p + q)
    def objective(params):
        eps = arma_model(params, y, p, q)
        return np.sum(eps ** 2)
    res = minimize(objective, init_params, method='BFGS')
    fitted_params = res.x
    residuals = arma_model(fitted_params, y, p, q)
    return fitted_params, residuals

params, eps = fit_arma_manual(y, p, q)
print("\nFitted parameters:", params)

def print_arma_equation(params, p, q):
    eq = f"y_t = {params[0]:.4f}"
    for i in range(p):
        eq += f" + ({params[i+1]:.4f})·y_(t-{i+1})"
    for j in range(q):
        eq += f" + ({params[p+1+j]:.4f})·ε_(t-{j+1})"
    eq += " + ε_t"
    print("\n📘 Estimated ARMA Equation:")
    print(eq)

print_arma_equation(params, p, q)

def fit_arima_manual(y, p, d, q):
    y_diff = np.diff(y, n=d) if d > 0 else y
    init_params = np.zeros(1 + p + q)
    def objective(params):
        eps = arma_model(params, y_diff, p, q)
        return np.sum(eps ** 2)
    res = minimize(objective, init_params, method='BFGS')
    fitted_params = res.x
    residuals = arma_model(fitted_params, y_diff, p, q)
    return fitted_params, residuals, y_diff

d=1
params, eps, y_diff = fit_arima_manual(y, p, d, q)
print("\nFitted parameters:", params)

def print_arima_equation(params, p, d, q):
    eq = f"(1 - B)^{d} y_t = {params[0]:.4f}"
    for i in range(p):
        eq += f" + ({params[i+1]:.4f})·y_(t-{i+1})"
    for j in range(q):
        eq += f" + ({params[p+1+j]:.4f})·ε_(t-{j+1})"
    eq += " + ε_t"
    print("\n📘 Estimated ARIMA Equation:")
    print(eq)

print_arima_equation(params, p, d, q)


res_acf = acf(eps[max(p, q):], 20)
mu, sigma = np.mean(eps), np.std(eps)

plt.figure(figsize=(14,5))
plt.plot(eps, label='Residuals')
plt.title("Residuals over time")
plt.legend()
plt.tight_layout()
plt.show()

# Z-test visualization
z_scores = (eps - mu) / sigma
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(7,4))
plt.hist(z_scores, bins=20, density=True, alpha=0.6, label='Residual z-scores')
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
plt.title("Z-Test Curve (Residuals vs Normal)")
plt.legend(); plt.show()

z_stat = (np.mean(eps)) / (np.std(eps) / np.sqrt(len(eps)))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"\nZ-test for residual mean=0 → z={z_stat:.3f}, p={p_value:.3f}")
