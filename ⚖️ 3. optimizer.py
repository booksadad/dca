import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf 
from factors import sigmoid

def run_black_litterman(port_df, returns_1y, dynamic_lambda, turnover_penalty, max_sector_cap, current_weights_arr):
    port_tickers = port_df['Ticker'].tolist()
    port_returns = returns_1y[port_tickers]
    num_assets = len(port_tickers)
    
    lw = LedoitWolf()
    lw.fit(port_returns)
    shrunk_cov = lw.covariance_ * 252 
    
    inv_vol = 1.0 / returns_1y[port_tickers].std()
    w_eq = (inv_vol / inv_vol.sum()).values
    Pi = dynamic_lambda * np.dot(shrunk_cov, w_eq) 
    
    tau = 0.05
    Q = np.zeros(num_assets)
    omega_diag = np.zeros(num_assets)
    P = np.eye(num_assets) 
    
    for i, t in enumerate(port_tickers):
        alpha_score = port_df.loc[port_df['Ticker'] == t, 'Alpha_Score'].values[0]
        conviction = sigmoid(alpha_score) 
        signal_strength = (conviction * 0.25) - 0.10 
        Q[i] = Pi[i] + signal_strength
        
        base_uncertainty = shrunk_cov[i, i] * tau
        omega_diag[i] = base_uncertainty / max(conviction, 0.01)

    Omega = np.diag(omega_diag)
    tau_cov_inv = np.linalg.inv(tau * shrunk_cov)
    omega_inv = np.linalg.inv(Omega)
    
    term1 = np.linalg.inv(tau_cov_inv + np.dot(np.dot(P.T, omega_inv), P))
    term2 = np.dot(tau_cov_inv, Pi) + np.dot(np.dot(P.T, omega_inv), Q)
    mu_bl = np.dot(term1, term2) 
    
    def neg_utility(w):
        expected_return = np.sum(mu_bl * w)
        portfolio_variance = np.dot(w.T, np.dot(shrunk_cov, w)) 
        return -(expected_return - (dynamic_lambda / 2.0) * portfolio_variance - (0.5 * np.sum(w**2)) - (turnover_penalty * np.sum(np.abs(w - current_weights_arr))))
        
    max_single_weight = 0.30 
    bounds = tuple((0.0, max_single_weight) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    max_turnover_cap = 0.60 
    constraints.append({'type': 'ineq', 'fun': lambda w: max_turnover_cap - np.sum(np.abs(w - current_weights_arr))})
    
    unique_sectors = port_df['Sector'].unique()
    for sector in unique_sectors:
        sec_indices = [i for i, t in enumerate(port_tickers) if port_df.loc[port_df['Ticker'] == t, 'Sector'].values[0] == sector]
        constraints.append({'type': 'ineq', 'fun': lambda w, idx=sec_indices: max_sector_cap - np.sum(w[idx])})
    
    opt_result = minimize(neg_utility, num_assets * [1./num_assets], method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 300})
    
    return opt_result
