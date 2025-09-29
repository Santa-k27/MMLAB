import random
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest



# Linear regression: y = a + bx
def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    # Slope (b) and intercept (a)
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    a = (sum_y - b * sum_x) / n
    return a, b

# Quadratic regression: y = a + bx + cx^2
def quadratic_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_x2 = sum(xi * xi for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2y = sum(xi * xi * yi for xi, yi in zip(x, y))

    # Solve system of equations: [n, sum_x, sum_x2][a, b, c] = [sum_y, sum_xy, sum_x2y]
    denom = n * (sum_x2 * sum_x4 - sum_x3 * sum_x3) - sum_x * (sum_x * sum_x4 - sum_x2 * sum_x3) + sum_x2 * (sum_x * sum_x3 - sum_x2 * sum_x2)
    a = (sum_y * (sum_x2 * sum_x4 - sum_x3 * sum_x3) - sum_x * (sum_xy * sum_x4 - sum_x2y * sum_x3) + sum_x2 * (sum_xy * sum_x3 - sum_x2y * sum_x2)) / denom
    b = (n * (sum_xy * sum_x4 - sum_x2y * sum_x3) - sum_y * (sum_x * sum_x4 - sum_x2 * sum_x3) + sum_x2 * (sum_x * sum_x2y - sum_x2 * sum_xy)) / denom
    c = (n * (sum_x2 * sum_x2y - sum_x3 * sum_xy) - sum_x * (sum_x * sum_x2y - sum_x2 * sum_xy) + sum_y * (sum_x * sum_x3 - sum_x2 * sum_x2)) / denom
    return a, b, c

# Cubic regression: y = a + bx + cx^2 + dx^3
def cubic_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_x2 = sum(xi * xi for xi in x)
    sum_x3 = sum(xi**3 for xi in x)
    sum_x4 = sum(xi**4 for xi in x)
    sum_x5 = sum(xi**5 for xi in x)
    sum_x6 = sum(xi**6 for xi in x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2y = sum(xi * xi * yi for xi, yi in zip(x, y))
    sum_x3y = sum(xi**3 * yi for xi, yi in zip(x, y))

    # Solve system of equations (4x4 matrix)
    A = [
        [n, sum_x, sum_x2, sum_x3],
        [sum_x, sum_x2, sum_x3, sum_x4],
        [sum_x2, sum_x3, sum_x4, sum_x5],
        [sum_x3, sum_x4, sum_x5, sum_x6]
    ]
    B = [sum_y, sum_xy, sum_x2y, sum_x3y]

    # Gauss-Jordan elimination
    for i in range(4):
        pivot = A[i][i]
        for j in range(4):
            A[i][j] /= pivot
        B[i] /= pivot
        for k in range(4):
            if k != i:
                factor = A[k][i]
                for j in range(4):
                    A[k][j] -= factor * A[i][j]
                B[k] -= factor * B[i]

    return B  # [a, b, c, d]

# Calculate predicted values
def predict_linear(x, a, b):
    return [a + b * xi for xi in x]

def predict_quadratic(x, a, b, c):
    return [a + b * xi + c * xi * xi for xi in x]

def predict_cubic(x, a, b, c, d):
    return [a + b * xi + c * xi * xi + d * xi**3 for xi in x]

# Calculate R-squared
def r_squared(y, y_pred):
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean)**2 for yi in y)
    ss_res = sum((yi - y_pred_i)**2 for yi, y_pred_i in zip(y, y_pred))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

# F-test for model comparison
def f_test(y, y_pred1, y_pred2, p1, p2):
    n = len(y)
    ss_res1 = sum((yi - y_pred_i)**2 for yi, y_pred_i in zip(y, y_pred1))
    ss_res2 = sum((yi - y_pred_i)**2 for yi, y_pred_i in zip(y, y_pred2))
    df1 = n - p1
    df2 = n - p2
    f_stat = ((ss_res1 - ss_res2) / (p2 - p1)) / (ss_res2 / df2) if ss_res2 != 0 and df2 != 0 else float('inf')
    return f_stat


def auto_cov_fn(data,lag):
  mean=np.mean(data)
  n=len(data)
  num=0
  for i in range(n-lag):
    num+=((data[i]-mean)*(data[i+lag]-mean))
  return num/(n)

rho_k=[]

n_lags=25
plt.figure(figsize=(10, 6))
for i in range(n_lags+1):
  rho=(auto_cov_fn(mdf['PROR'],i))/(auto_cov_fn(mdf['PROR'],0))
  rho_k.append(float(rho))
plt.stem(rho_k)
plt.title('Autocorrelation Function (ACF) for monthly data of PVR STOCK')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)


#plot acf using plot_ACF function
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(mdf['PROR'],lags=25)
plt.show()


def partial_auto_cov_fn(data, lag):
    """Calculates the Partial Autocorrelation Function (PACF) manually using Yule-Walker equations."""
    acf_vals = [auto_cov_fn(data, i) / auto_cov_fn(data, 0) for i in range(lag + 1)]
    if lag == 0:
        return 1.0
    elif lag == 1:
        return acf_vals[1]
    else:
        # Construct the matrix and vector for the Yule-Walker equations: rho(j) = sum(i=0 to k) (phi(ik)* rho(j-i)) for j = 1,2,3...k
        # We are solving for phi_kk (the PACF at lag k) and the intermediate phi_ki (i < k)

        matrix_A = np.zeros((lag, lag))
        vector_b = np.zeros(lag)

        for i in range(lag):
            vector_b[i] = acf_vals[i + 1]
            for j in range(lag):
                matrix_A[i, j] = acf_vals[abs(i - j)]

        phi_coeffs = np.linalg.solve(matrix_A, vector_b)
        return phi_coeffs[-1]

n_lags = 25
pacf_monthly = [partial_auto_cov_fn(mdf['PROR'], lag) for lag in range(n_lags + 1)]

# Plot PACF for monthly data
plt.figure(figsize=(10, 6))
plt.stem(range(n_lags + 1), pacf_monthly)
plt.title('Partial Autocorrelation Function (PACF) for monthly data of PVR STOCK (Manual)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.grid(True)
plt.show()

#plot pacf using plot_pacf
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(mdf['PROR'],lags=25)
plt.show()


def second_order_exponential_smoothening(data, alpha):
    first_order = [data[0]]
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * (first_order[-1])
        first_order.append(smoothed_value)
    second_order = [first_order[0]]
    for i in range(1, len(first_order)):
        smoothed_value = alpha * first_order[i] + (1 - alpha) * (second_order[-1])
        second_order.append(smoothed_value)
    res=[]
    for i in range(len(data)):
      res.append(2*first_order[i]-second_order[i])

    return np.array(first_order), np.array(second_order),res

for i, alpha in enumerate(lam1):
    first_order, second_order, res = second_order_exponential_smoothening(mdf['PROR'].values, alpha)
    plt.figure(figsize=(12, 8))
    plt.plot(mdf['PROR'].values, label='Original Data', color='gray', linestyle='--')
    plt.plot(first_order, label=f'first order Data for lambda{i+1} = ({alpha})')
    plt.plot(second_order, label=f'second order Data for lambda{i+1} = ({alpha})')
    plt.plot(res, label=f'optimal y_cap Data for lambda{i+1} = ({alpha})')
    plt.title(f'Second-Order Exponential Smoothing with lambda{i+1} = {alpha}')
    plt.xlabel('Time')
    plt.ylabel('PROR')
    plt.legend()
    plt.grid(True)
    plt.show()

    z_statistic, p_value = ztest(mdf['PROR'].values, res)

    print(f"Z-test results for lambda{i+1} = {alpha}:")
    print(f"  Z-statistic: {z_statistic}")
    print(f"  P-value: {p_value}")

    # Interpret the results (common alpha level is 0.05)
    alpha = 0.05
    if p_value < alpha:
      print(f"  Reject the null hypothesis: There is a significant difference between the original data and the smoothed data for {key}.")
    else:
      print(f"  Fail to reject the null hypothesis: There is no significant difference between the original data and the smoothed data for {key}.")
    print("-" * 30)