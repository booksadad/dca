import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def sigmoid_conviction(score):
    """แปลง Alpha Score เป็นค่าความมั่นใจ (0 ถึง 1) ผ่าน Sigmoid"""
    return 1 / (1 + np.exp(-score))

def run_institutional_black_litterman(
    port_df, 
    returns_df, 
    regime_probs, # รับค่า P_BULL, P_PANIC จาก Layer 3
    current_weights, 
    base_tau=0.05, 
    risk_aversion=2.5, 
    turnover_limit=0.20,
    max_sector_cap=0.30
):
    """
    [Layer 4] Advanced Black-Litterman with Cost-Aware Optimization
    """
    num_assets = len(port_df)
    tickers = port_df['Ticker'].tolist()
    asset_returns = returns_df[tickers].fillna(0)
    
    # 1. Ledoit-Wolf Shrinkage Covariance (ลด Estimation Error)
    lw = LedoitWolf()
    lw.fit(asset_returns)
    Sigma = lw.covariance_ * 252
    
    # 2. Market Implied Equilibrium Returns (Pi)
    inv_vol = 1.0 / np.std(asset_returns, axis=0)
    w_mkt = (inv_vol / np.sum(inv_vol)).values # สมมติฐาน Market Cap Weight ผ่าน Inverse Vol
    Pi = risk_aversion * np.dot(Sigma, w_mkt)
    
    # 3. Dynamic Tau (ปรับตาม HMM Regime)
    p_panic = regime_probs.get('P_PANIC', 0.0)
    tau = base_tau * (1 - (0.5 * p_panic))
    
    # 4. สร้าง Views (P, Q, Omega)
    P = np.eye(num_assets)
    Q = np.zeros(num_assets)
    omega_diag = np.zeros(num_assets)
    
    ic_estimate = 0.05 # Signal Strength
    
    for i, t in enumerate(tickers):
        alpha_score = port_df.loc[port_df['Ticker'] == t, 'Alpha_Score'].values[0]
        asset_vol = np.sqrt(Sigma[i, i])
        
        # Sigmoid Conviction
        conviction = sigmoid_conviction(alpha_score)
        
        # Excess return view Q
        Q[i] = Pi[i] + (conviction * ic_estimate * asset_vol * alpha_score)
        
        # Omega (Uncertainty) - ยิ่ง Conviction สูง Omega ยิ่งต่ำ
        omega_diag[i] = (tau * Sigma[i, i]) / max(conviction, 0.01)

    Omega = np.diag(omega_diag)
    
    # 5. แก้สมการ Bayesian Black-Litterman (Posterior E[R])
    tau_cov_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(Omega)
    
    term1 = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
    term2 = (tau_cov_inv @ Pi) + (P.T @ omega_inv @ Q)
    mu_bl = term1 @ term2 

    # 6. Cost-Aware Objective Function (L2 Penalty + Txn Cost)
    gamma_l2 = 0.10 # L2 Penalty Weight
    txn_cost_bps = 0.0020 # สมมติค่าคอม+Slippage รวม 0.20%
    
    def objective(w):
        expected_ret = np.dot(mu_bl, w)
        port_var = np.dot(w.T, np.dot(Sigma, w))
        
        # Penalty 1: L2 Norm (ป้องกันน้ำหนักกระจุกตัว)
        l2_penalty = gamma_l2 * np.sum(w**2)
        
        # Penalty 2: Transaction Cost (หัก gross return ทิ้ง)
        turnover = np.sum(np.abs(w - current_weights))
        cost_penalty = turnover * txn_cost_bps
        
        # เป้าหมายคือ Minimize Negative Utility (Maximize Utility)
        utility = expected_ret - (risk_aversion / 2.0 * port_var) - l2_penalty - cost_penalty
        return -utility

    # 7. Constraints Setup
    bounds = tuple((0.0, 0.15) for _ in range(num_assets)) # Stock Limit <= 15% (ผ่อนปรนให้พอร์ตเล็ก)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, # น้ำหนักรวม = 100%
        {'type': 'ineq', 'fun': lambda w: turnover_limit - np.sum(np.abs(w - current_weights))} # Turnover Limit
    ]
    
    # Sector Constraint (<= 30%)
    for sector in port_df['Sector'].unique():
        sec_idx = [i for i, t in enumerate(tickers) if port_df.loc[port_df['Ticker'] == t, 'Sector'].values[0] == sector]
        constraints.append({'type': 'ineq', 'fun': lambda w, idx=sec_idx: max_sector_cap - np.sum(w[idx])})
    
    # 8. Run SLSQP Optimizer
    # ใช้ current_weights เป็นจุดตั้งต้น เพื่อให้ Optimizer ขยับตัวน้อยที่สุด
    initial_guess = current_weights if np.sum(current_weights) > 0.9 else (np.ones(num_assets) / num_assets)
    
    opt_result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500})
    
    return opt_result, mu_bl
