import os
import warnings
import numpy as np
import pandas as pd

from .utils import init_debug_dir, csvwrite_debug

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def decorator(func):
            return func
        return decorator

_HAS_RUN = False

def reset_kf_run_flag():
    """Resets the debug export lock for run_kf, allowing it to export files on the next call."""
    global _HAS_RUN
    _HAS_RUN = False

@njit
def numba_pinv(A):
    U, S, Vt = np.linalg.svd(A)
    S_inv = np.zeros(len(S))
    for i in range(len(S)):
        if S[i] > 1e-12:
            S_inv[i] = 1.0 / S[i]
    return Vt.T @ np.diag(S_inv) @ U.T

@njit
def _skf_jit(Y, Z, R_mat, T_mat, Q_mat, A_0, P_0):
    n, m = Z.shape
    nobs = Y.shape[1]
    
    Am = np.full((m, nobs), np.nan)
    Pm = np.full((m, m, nobs), np.nan)
    AmU = np.full((m, nobs + 1), np.nan)
    PmU = np.full((m, m, nobs + 1), np.nan)
    loglik = 0.0
    
    Au = A_0.copy().reshape(-1, 1)
    Pu = P_0.copy()
    
    AmU[:, 0] = Au.flatten()
    PmU[:, :, 0] = Pu
    
    PZF = np.zeros((m, n))
    Z_t = np.zeros((n, m))
    
    for t in range(nobs):
        A_pred = T_mat @ Au
        P_pred = T_mat @ Pu @ T_mat.T + Q_mat
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        y_t = Y[:, t:t+1]
        
        # Identify non-NaN indices
        ix_list = []
        for i in range(n):
            if not np.isnan(y_t[i, 0]):
                ix_list.append(i)
                
        if len(ix_list) == 0:
            Au = A_pred
            Pu = P_pred
        else:
            y_obs = np.zeros((len(ix_list), 1))
            Z_obs = np.zeros((len(ix_list), m))
            R_obs = np.zeros((len(ix_list), len(ix_list)))
            for idx, i in enumerate(ix_list):
                y_obs[idx, 0] = y_t[i, 0]
                Z_obs[idx, :] = Z[i, :]
            for idx1, i1 in enumerate(ix_list):
                for idx2, i2 in enumerate(ix_list):
                    R_obs[idx1, idx2] = R_mat[i1, i2]
                    
            PZ = P_pred @ Z_obs.T
            F_inv = np.linalg.inv(Z_obs @ PZ + R_obs)
            PZF_t = PZ @ F_inv
            V = y_obs - Z_obs @ A_pred
            Au = A_pred + PZF_t @ V
            Pu = P_pred - PZF_t @ PZ.T
            Pu = 0.5 * (Pu + Pu.T)
            
            sign, logdet = np.linalg.slogdet(F_inv)
            loglik += 0.5 * (logdet - (V.T @ F_inv @ V)[0, 0])
            
            if t == nobs - 1:
                PZF = PZF_t
                Z_t = Z_obs
                
        Am[:, t] = A_pred.flatten()
        Pm[:, :, t] = P_pred
        AmU[:, t + 1] = Au.flatten()
        PmU[:, :, t + 1] = Pu
        
    KZ = np.zeros((m, m))
    if len(ix_list) > 0:
        KZ = PZF @ Z_t
        
    return Am, Pm, AmU, PmU, KZ, loglik

@njit
def _fis_jit(T_mat, Am, Pm, AmU, PmU, KZ):
    m, nobs = Am.shape
    AmT = np.zeros((m, nobs + 1))
    PmT = np.zeros((m, m, nobs + 1))
    PmT_1 = np.zeros((m, m, nobs))
    
    AmT[:, nobs] = AmU[:, nobs]
    PmT[:, :, nobs] = PmU[:, :, nobs]
    
    PmT_1[:, :, nobs - 1] = (np.eye(m) - KZ) @ T_mat @ PmU[:, :, nobs - 1]
    J_2 = PmU[:, :, nobs - 1] @ T_mat.T @ numba_pinv(Pm[:, :, nobs - 1])
    
    for t in range(nobs - 1, -1, -1):
        PmU_t = PmU[:, :, t]
        Pm1 = Pm[:, :, t]
        P_T = PmT[:, :, t + 1]
        P_T1 = PmT_1[:, :, t]
        
        J_1 = J_2
        AmT[:, t] = AmU[:, t] + J_1 @ (AmT[:, t + 1] - T_mat @ AmU[:, t])
        PmT[:, :, t] = PmU_t + J_1 @ (P_T - Pm1) @ J_1.T
        
        if t > 0:
            J_2 = PmU[:, :, t - 1] @ T_mat.T @ numba_pinv(Pm[:, :, t - 1])
            PmT_1[:, :, t - 1] = PmU_t @ J_2.T + J_1 @ (P_T1 - T_mat @ PmU_t) @ J_2.T
            
    return AmT, PmT, PmT_1

def run_kf(y, A, C, Q, R, x_0, Sig_0, debug_dir=None):
    """
    Kalman Filter and Smoother (Sujit, 1999).
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    global _HAS_RUN

    target_dir = init_debug_dir('runKF', debug_dir=debug_dir)

    if _HAS_NUMBA:
        Am, Pm, AmU, PmU, KZ, loglik = _skf_jit(y, C, R, A, Q, x_0, Sig_0)
        xsmooth, Vsmooth, VVsmooth = _fis_jit(A, Am, Pm, AmU, PmU, KZ)
    else:
        def miss_data(y_vec, C_mat, R_mat):
            ix = ~np.isnan(y_vec).flatten()
            e = np.eye(y_vec.shape[0])
            L = e[:, ix]
            y_out = y_vec[ix].reshape(-1, 1)
            C_out = C_mat[ix, :]
            R_out = R_mat[ix][:, ix]
            return y_out, C_out, R_out, L

        def skf(Y, Z, R_mat, T_mat, Q_mat, A_0, P_0):
            n, m = Z.shape
            nobs = Y.shape[1]
            
            S = {
                'Am': np.full((m, nobs), np.nan),
                'Pm': np.full((m, m, nobs), np.nan),
                'AmU': np.full((m, nobs + 1), np.nan),
                'PmU': np.full((m, m, nobs + 1), np.nan),
                'loglik': 0.0
            }
            
            Au = A_0.copy().reshape(-1, 1)
            Pu = P_0.copy()
            
            S['AmU'][:, 0] = Au.flatten()
            S['PmU'][:, :, 0] = Pu
            
            y_t = np.array([])
            PZF = np.zeros((m, n))
            Z_t = np.zeros((n, m))
            
            for t in range(nobs):
                A_pred = T_mat @ Au
                P_pred = T_mat @ Pu @ T_mat.T + Q_mat
                P_pred = 0.5 * (P_pred + P_pred.T)
                
                y_t, Z_t, R_t, _ = miss_data(Y[:, t:t+1], Z, R_mat)
                
                if y_t.size == 0:
                    Au = A_pred
                    Pu = P_pred
                else:
                    PZ = P_pred @ Z_t.T
                    iF = np.linalg.inv(Z_t @ PZ + R_t)
                    PZF = PZ @ iF
                    V = y_t - Z_t @ A_pred
                    Au = A_pred + PZF @ V
                    Pu = P_pred - PZF @ PZ.T
                    Pu = 0.5 * (Pu + Pu.T)
                    sign, logdet = np.linalg.slogdet(iF)
                    S['loglik'] += 0.5 * (logdet - (V.T @ iF @ V).item())
                    
                S['Am'][:, t] = A_pred.flatten()
                S['Pm'][:, :, t] = P_pred
                S['AmU'][:, t + 1] = Au.flatten()
                S['PmU'][:, :, t + 1] = Pu
                
            S['KZ'] = np.zeros((m, m)) if y_t.size == 0 else PZF @ Z_t
            return S

        def fis(Y, Z, R_mat, T_mat, Q_mat, S):
            m, nobs = S['Am'].shape
            S['AmT'] = np.zeros((m, nobs + 1))
            S['PmT'] = np.zeros((m, m, nobs + 1))
            S['PmT_1'] = np.zeros((m, m, nobs))
            
            S['AmT'][:, nobs] = S['AmU'][:, nobs]
            S['PmT'][:, :, nobs] = S['PmU'][:, :, nobs]
            
            S['PmT_1'][:, :, nobs - 1] = (np.eye(m) - S['KZ']) @ T_mat @ S['PmU'][:, :, nobs - 1]
            J_2 = S['PmU'][:, :, nobs - 1] @ T_mat.T @ np.linalg.pinv(S['Pm'][:, :, nobs - 1])
            
            for t in range(nobs - 1, -1, -1):
                PmU = S['PmU'][:, :, t]
                Pm1 = S['Pm'][:, :, t]
                P_T = S['PmT'][:, :, t + 1]
                P_T1 = S['PmT_1'][:, :, t]
                
                J_1 = J_2
                S['AmT'][:, t] = S['AmU'][:, t] + J_1 @ (S['AmT'][:, t + 1] - T_mat @ S['AmU'][:, t])
                S['PmT'][:, :, t] = PmU + J_1 @ (P_T - Pm1) @ J_1.T
                
                if t > 0:
                    J_2 = S['PmU'][:, :, t - 1] @ T_mat.T @ np.linalg.pinv(S['Pm'][:, :, t - 1])
                    S['PmT_1'][:, :, t - 1] = PmU @ J_2.T + J_1 @ (P_T1 - T_mat @ PmU) @ J_2.T
                    
            return S

        S = skf(y, C, R, A, Q, x_0, Sig_0)
        S = fis(y, C, R, A, Q, S)
        xsmooth = S['AmT']
        Vsmooth = S['PmT']
        VVsmooth = S['PmT_1']
        loglik = S['loglik']

    if debug_dir is not None and not _HAS_RUN:
        m1, m2, nobs_plus_1 = Vsmooth.shape
        nobs = VVsmooth.shape[2]
        Vsmooth_flat = Vsmooth.reshape((m1 * m2, nobs_plus_1), order='F')
        VVsmooth_flat = VVsmooth.reshape((m1 * m2, nobs), order='F')
        
        csvwrite_debug(target_dir, 'xsmooth.csv', xsmooth)
        csvwrite_debug(target_dir, 'Vsmooth_flat.csv', Vsmooth_flat)
        csvwrite_debug(target_dir, 'VVsmooth_flat.csv', VVsmooth_flat)
        csvwrite_debug(target_dir, 'loglik.csv', loglik)
        _HAS_RUN = True

    return xsmooth, Vsmooth, VVsmooth, loglik

def run_kalman_smoother(X, P_struct, lag, debug_dir=None):
    """
    Kalman Smoother with lags for news calculation.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    target_dir = init_debug_dir('para_constDG', debug_dir=debug_dir)

    Z_0, V_0 = P_struct['Z_0'], P_struct['V_0']
    A, C = P_struct['A'], P_struct['C']
    Q, R = P_struct['Q'], P_struct['R']
    Mx, Wx = P_struct['Mx'], P_struct['Wx']

    from dfm_nowcast.data import DFMPreprocessor
    preprocessor = DFMPreprocessor()
    preprocessor.Mx = Mx
    preprocessor.Wx = Wx
    xNaN = preprocessor.transform(X)
    y = xNaN.T

    if _HAS_NUMBA:
        Am, Pm, AmU, PmU, KZ, loglik = _skf_jit(y, C, R, A, Q, Z_0, V_0)
        xsmooth, Vsmooth, VVsmooth = _fis_jit(A, Am, Pm, AmU, PmU, KZ)
        Ps = Vsmooth[:, :, 1:]
        Pf = PmU[:, :, 1:]
        Zsmooth = xsmooth.T
    else:
        def miss_data(y_vec, C_mat, R_mat):
            ix = ~np.isnan(y_vec).flatten()
            e = np.eye(y_vec.shape[0])
            L = e[:, ix]
            return y_vec[ix].reshape(-1, 1), C_mat[ix, :], R_mat[ix][:, ix], L

        def skf(Y, Z, R_mat, T_mat, Q_mat, A_0, P_0):
            n, m = Z.shape
            nobs = Y.shape[1]
            
            S = {
                'Am': np.full((m, nobs), np.nan), 'Pm': np.full((m, m, nobs), np.nan),
                'AmU': np.full((m, nobs + 1), np.nan), 'PmU': np.full((m, m, nobs + 1), np.nan),
                'loglik': 0.0
            }
            
            Au = A_0.copy().reshape(-1, 1)
            Pu = P_0.copy()
            
            S['AmU'][:, 0] = Au.flatten()
            S['PmU'][:, :, 0] = Pu
            y_t = np.array([])
            PZF = np.zeros((m, n))
            Z_t = np.zeros((n, m))
            
            for t in range(nobs):
                A_pred = T_mat @ Au
                P_pred = T_mat @ Pu @ T_mat.T + Q_mat
                P_pred = 0.5 * (P_pred + P_pred.T)
                
                y_t, Z_t, R_t, _ = miss_data(Y[:, t:t+1], Z, R_mat)
                
                if y_t.size == 0:
                    Au = A_pred
                    Pu = P_pred
                else:
                    PZ = P_pred @ Z_t.T
                    iF = np.linalg.inv(Z_t @ PZ + R_t)
                    PZF = PZ @ iF
                    V = y_t - Z_t @ A_pred
                    Au = A_pred + PZF @ V
                    Pu = P_pred - PZF @ PZ.T
                    Pu = 0.5 * (Pu + Pu.T)
                    sign, logdet = np.linalg.slogdet(iF)
                    S['loglik'] += 0.5 * (logdet - (V.T @ iF @ V).item())
                    
                S['Am'][:, t] = A_pred.flatten()
                S['Pm'][:, :, t] = P_pred
                S['AmU'][:, t + 1] = Au.flatten()
                S['PmU'][:, :, t + 1] = Pu
                
            S['KZ'] = np.zeros((m, m)) if y_t.size == 0 else PZF @ Z_t
            return S

        def fis(Y, Z, R_mat, T_mat, Q_mat, S):
            m, nobs = S['Am'].shape
            S['AmT'] = np.zeros((m, nobs + 1))
            S['PmT'] = np.zeros((m, m, nobs + 1))
            S['PmT_1'] = np.zeros((m, m, nobs))
            
            S['AmT'][:, nobs] = S['AmU'][:, nobs]
            S['PmT'][:, :, nobs] = S['PmU'][:, :, nobs]
            
            S['PmT_1'][:, :, nobs - 1] = (np.eye(m) - S['KZ']) @ T_mat @ S['PmU'][:, :, nobs - 1]
            J_2 = S['PmU'][:, :, nobs - 1] @ T_mat.T @ np.linalg.pinv(S['Pm'][:, :, nobs - 1])
            
            for t in range(nobs - 1, -1, -1):
                PmU = S['PmU'][:, :, t]
                Pm1 = S['Pm'][:, :, t]
                P_T = S['PmT'][:, :, t + 1]
                P_T1 = S['PmT_1'][:, :, t]
                
                J_1 = J_2
                S['AmT'][:, t] = S['AmU'][:, t] + J_1 @ (S['AmT'][:, t + 1] - T_mat @ S['AmU'][:, t])
                S['PmT'][:, :, t] = PmU + J_1 @ (P_T - Pm1) @ J_1.T
                
                if t > 0:
                    J_2 = S['PmU'][:, :, t - 1] @ T_mat.T @ np.linalg.pinv(S['Pm'][:, :, t - 1])
                    S['PmT_1'][:, :, t - 1] = PmU @ J_2.T + J_1 @ (P_T1 - T_mat @ PmU) @ J_2.T
                    
            return S

        Sf = skf(y, C, R, A, Q, Z_0, V_0)
        Ss = fis(y, C, R, A, Q, Sf)
        Ps = Ss['PmT'][:, :, 1:]
        Pf = Sf['PmU'][:, :, 1:]
        Zsmooth = Ss['AmT'].T
        Vsmooth = Ss['PmT']

    Plag = [Ps]

    for jk in range(1, lag + 1):
        next_Plag = np.zeros_like(Ps)
        for jt in range(Ps.shape[2] - 1, lag - 1, -1):
            Pf_lag = Pf[:, :, jt - jk]
            As = Pf_lag @ A.T @ np.linalg.pinv(A @ Pf_lag @ A.T + Q)
            next_Plag[:, :, jt] = As @ Plag[jk - 1][:, :, jt]
        Plag.append(next_Plag)

    x_sm = Zsmooth[1:, :] @ C.T
    X_sm = preprocessor.inverse_transform(x_sm)

    Res = {
        'Plag': Plag,
        'P': Vsmooth,
        'X_sm': X_sm
    }

    csvwrite_debug(target_dir, 'X_sm.csv', Res['X_sm'])
    m1, m2, nobs_plus_1 = Vsmooth.shape
    Vsmooth_flat = Vsmooth.reshape((m1 * m2, nobs_plus_1), order='F')
    csvwrite_debug(target_dir, 'P_flat.csv', Vsmooth_flat)

    return Res

# Legacy alias
para_constdg = run_kalman_smoother
