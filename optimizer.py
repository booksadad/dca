import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def run_institutional_black_litterman(
    port_df, returns_df, regime_probs, current_weights, 
    base_tau=0.05, risk_aversion=2.5, turnover_limit=0.50, max_sector_cap=0.50
):
    num_assets = len(port_df)
    tickers = port_df['Ticker'].tolist()
    asset_returns = returns_df[tickers].fillna(0)
    
    lw = LedoitWolf()
    lw.fit(asset_returns)
    Sigma = lw.covariance_ * 252
    
    # 🛠️ THE FIX: เลี่ยงการใช้ In-place division (/=) บน Numpy Read-only View
    # ให้ Pandas คำนวณสัดส่วนให้เสร็จก่อน แล้วค่อยดึง .values ออกมาทีหลังสุด
    inv_vol = 1.0 / (np.std(asset_returns, axis=0) + 1e-9)
    w_mkt = (inv_vol / inv_vol.sum()).values
    
    Pi = risk_aversion * np.dot(Sigma, w_mkt)
    
    tau = base_tau * (1.0 - (0.5 * regime_probs.get('P_PANIC', 0.0)))
    P, Q, omega_diag = np.eye(num_assets), np.zeros(num_assets), np.zeros(num_assets)
    
    for i, t in enumerate(tickers):
        score = port_df.loc[port_df['Ticker'] == t, 'Alpha_Score'].values[0]
        conviction = 1.0 / (1.0 + np.exp(-score))
        Q[i] = Pi[i] + (conviction * 0.05 * np.sqrt(Sigma[i, i]) * score)
        omega_diag[i] = (tau * Sigma[i, i]) / max(conviction, 0.01)

    Omega = np.diag(omega_diag)
    tau_cov_inv, omega_inv = np.linalg.inv(tau * Sigma), np.linalg.inv(Omega)
    term1 = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
    mu_bl = term1 @ ((tau_cov_inv @ Pi) + (P.T @ omega_inv @ Q))
    
    def objective(w):
        utility = np.dot(mu_bl, w) - (risk_aversion / 2.0 * np.dot(w.T, np.dot(Sigma, w)))
        utility -= 0.10 * np.sum(w**2) + np.sum(np.abs(w - current_weights)) * 0.0020
        return -utility

    bounds = tuple((0.0, 0.45) for _ in range(num_assets))
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: turnover_limit - np.sum(np.abs(w - current_weights))}
    ]
    for sector in port_df['Sector'].unique():
        sec_idx = [i for i, t in enumerate(tickers) if port_df.loc[port_df['Ticker'] == t, 'Sector'].values[0] == sector]
        constraints.append({'type': 'ineq', 'fun': lambda w, idx=sec_idx: max_sector_cap - np.sum(w[idx])})
        
    res = minimize(
        objective, 
        current_weights if np.sum(current_weights) > 0.9 else (np.ones(num_assets)/num_assets), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'maxiter': 500}
    )
    return res, mu_bl
