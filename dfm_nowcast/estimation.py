import os
import warnings
import numpy as np
import scipy.linalg as la

from .utils import init_debug_dir, csvwrite_debug
from .kalman import run_kf
from .data import remnans_spline

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


def safe_inv(m):
    """
    Computes matrix inverse. Falls back to pseudo-inverse (pinv) if matrix is singular/near-singular.
    """
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        warnings.warn("Covariance matrix is singular or near-singular. Falling back to pseudo-inverse (pinv) to maintain numerical stability.", RuntimeWarning)
        return np.linalg.pinv(m)


def mixed_frequency_restrictions(growth_rate, debug_dir=None):
    """
    Replicates the MATLAB ML_MixedFrequencyRestrictions function.
    Sets up the state-space constraints for mixed-frequency (monthly/quarterly) data.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_MixedFrequencyRestrictions', debug_dir=debug_dir)

    if debug_dir is not None:
        with open(os.path.join(target_dir, 'GrowthRate.csv'), 'w') as f:
            f.write(f"{growth_rate}\n")

    # --- State-Space Constraint Setup --- #
    if growth_rate.lower() == 'qoq':
        r_constr = np.array([
            [2, -1,  0,  0,  0],
            [3,  0, -1,  0,  0],
            [2,  0,  0, -1,  0],
            [1,  0,  0,  0, -1]
        ], dtype=float)
        q = np.zeros((4, 1), dtype=float)
        
    elif growth_rate.lower() == 'yoy':
        ones_col = np.ones((2, 1), dtype=float)
        neg_eye = -np.eye(2, dtype=float)
        r_constr = np.hstack((ones_col, neg_eye))
        q = np.zeros((2, 1), dtype=float)
        
    else:
        raise ValueError('Invalid GrowthRate specified. Must be "qoq" or "yoy".')

    csvwrite_debug(target_dir, 'Rconstr.csv', r_constr)
    csvwrite_debug(target_dir, 'q.csv', q)

    return r_constr, q


# Legacy alias
ml_mixed_frequency_restrictions = mixed_frequency_restrictions


def idiosyncratic_law_of_motion(idiosyncratic, n_m, n_q, debug_dir=None):
    """
    Replicates the MATLAB ML_LawMotionIdiosyncratic function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_LawMotionIdiosyncratic', debug_dir=debug_dir)
    
    if debug_dir is not None:
        with open(os.path.join(target_dir, 'Idiosyncratic.csv'), 'w') as f:
            f.write(f"{idiosyncratic}\n")
        
    csvwrite_debug(target_dir, 'nM.csv', n_m)
    csvwrite_debug(target_dir, 'nQ.csv', n_q)

    # --- State-Space Constraint Setup --- #
    if idiosyncratic.lower() == 'autoregressive':
        i_idio = np.ones((int(n_m + n_q), 1), dtype=bool)
    elif idiosyncratic.lower() == 'whitenoise':
        i_idio = np.zeros((int(n_m + n_q), 1), dtype=bool)
    else:
        raise ValueError('Invalid Idiosyncratic specified. Must be "Autoregressive" or "WhiteNoise".')

    csvwrite_debug(target_dir, 'i_idio.csv', i_idio)

    return i_idio


# Legacy alias
ml_law_motion_idiosyncratic = idiosyncratic_law_of_motion


def evaluate_em_convergence(freq_estimation, curr_dates, prev_dates, t, debug_dir=None):
    """
    Replicates the MATLAB ML_EstimationYN function.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    target_dir = init_debug_dir('ML_EstimationYN', t=t, debug_dir=debug_dir)
    
    estimate = 0
    
    curr_month = np.array(curr_dates).flatten()[1]
    prev_month = np.array(prev_dates).flatten()[1]

    if freq_estimation.lower() == 'quarterly':
        if t == 0 or (curr_month == 1 and prev_month == 12) \
                  or (curr_month == 4 and prev_month == 3) \
                  or (curr_month == 7 and prev_month == 6) \
                  or (curr_month == 10 and prev_month == 9):
            estimate = 1
            
    elif freq_estimation.lower() == 'annual':
        if t == 0 or (curr_month == 1 and prev_month == 12):
            estimate = 1

    csvwrite_debug(target_dir, f'estimate_t{t+1}.csv', estimate)

    return estimate


# Legacy alias
ml_estimation_yn = evaluate_em_convergence


def fit_dfm_em(X, Par, Res_old=None, run_id=1, debug_dir=None):
    """
    EM algorithm for Dynamic Factor Models.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Intercept for MCMC estimator switch
    if Par.get('estimator', 'EM').upper() == 'MCMC':
        return fit_dfm_mcmc(X, Par, Res_old=Res_old)

    target_dir = init_debug_dir('EM_DFM', t=(0 if run_id == 1 else 1), debug_dir=debug_dir)

    from dfm_nowcast.data import DFMPreprocessor
    preprocessor = DFMPreprocessor(
        winsorization=Par.get('winsorization', False),
        winsorization_k=Par.get('winsorization_k', 4.0),
        robust_scaling=Par.get('robust_scaling', False)
    )
    if Res_old is None or not Res_old:
        xNaN = preprocessor.fit_transform(X)
        Mx = preprocessor.Mx
        Wx = preprocessor.Wx
    else:
        Mx = Res_old['Mx']
        Wx = Res_old['Wx']
        preprocessor.Mx = Mx
        preprocessor.Wx = Wx
        xNaN = preprocessor.transform(X)

    thresh = Par.get('thresh', 1e-3)
    r = np.array(Par['r'], dtype=int).flatten()
    p = int(Par['p'])
    max_iter = int(Par['max_iter'])
    i_idio = np.array(Par['i_idio'], dtype=bool).flatten()
    R_mat = np.array(Par['Rconstr'])
    q = np.array(Par['q'])
    nQ = int(Par['nQ'])
    blocks = np.array(Par['blocks'], dtype=bool)

    T, N = X.shape
    optNaN = {'method': 2, 'k': 3}
    y = xNaN.T

    if Res_old is None or not Res_old:
        A, C, Q, R, Z_0, V_0 = init_cond(xNaN, r, p, blocks, optNaN, R_mat, q, nQ, i_idio, debug_dir=debug_dir)
        
        csvwrite_debug(target_dir, 'Init_A.csv', A)
        csvwrite_debug(target_dir, 'Init_C.csv', C)
        csvwrite_debug(target_dir, 'Init_Q.csv', Q)
        csvwrite_debug(target_dir, 'Init_R.csv', R)
        csvwrite_debug(target_dir, 'Init_V_0.csv', V_0)
        
        previous_loglik = -np.inf
        num_iter = 0
        LL = [-np.inf]
        converged = False

        optNaN['method'] = 3
        y_est, _ = remnans_spline(xNaN, optNaN, debug_dir=debug_dir)
        y_est = y_est.T

        while (num_iter < max_iter) and not converged:
            C_new, R_new, A_new, Q_new, Z_0_new, V_0_new, loglik = em_step(
                y_est, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, debug_dir=debug_dir
            )
            
            C, R, A, Q = C_new, R_new, A_new, Q_new
            Z_0, V_0 = Z_0_new, V_0_new

            if num_iter > 2:
                converged, _ = em_converged(loglik, previous_loglik, thresh, True)
            
            LL.append(loglik)
            previous_loglik = loglik
            num_iter += 1
            
    else:
        A, C, Q, R = Res_old['A'], Res_old['C'], Res_old['Q'], Res_old['R']
        Z_0, V_0 = Res_old['Z_0'], Res_old['V_0']

    Zsmooth, Vsmooth, VVsmooth, _ = run_kf(y, A, C, Q, R, Z_0, V_0, debug_dir=debug_dir)
    Zsmooth_t = Zsmooth.T
    
    x_sm = Zsmooth_t[1:, :] @ C.T

    Res = {}
    Res['X_sm'] = preprocessor.inverse_transform(x_sm)
    Res['F'] = Zsmooth_t[1:, :]
    Res['C'], Res['R'], Res['A'], Res['Q'] = C, R, A, Q
    Res['Mx'], Res['Wx'] = Mx, Wx
    Res['Z_0'], Res['V_0'] = Z_0, V_0
    Res['r'], Res['p'] = r, p

    # Apply covariance regularization (Ridge shrinkage) if enabled
    if Par.get('covariance_regularization', False):
        lambda_val = Par.get('ridge_lambda', 1e-4)
        Res['R'] += lambda_val * np.eye(Res['R'].shape[0])

    csvwrite_debug(target_dir, 'Res_X_sm.csv', Res['X_sm'])
    csvwrite_debug(target_dir, 'Res_F.csv', Res['F'])
    csvwrite_debug(target_dir, 'Res_C.csv', Res['C'])
    csvwrite_debug(target_dir, 'Res_A.csv', Res['A'])
    csvwrite_debug(target_dir, 'Res_R.csv', Res['R'])
    csvwrite_debug(target_dir, 'Res_Q.csv', Res['Q'])
    csvwrite_debug(target_dir, 'Res_Z_0.csv', Res['Z_0'])
    csvwrite_debug(target_dir, 'Res_V_0.csv', Res['V_0'])
    if (Res_old is None or not Res_old) and debug_dir is not None:
        csvwrite_debug(target_dir, 'LL.csv', np.array(LL))

    return Res


def em_step(y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks, debug_dir=None):
    n, T = y.shape
    nM = n - nQ
    pC = R_mat.shape[1]
    ppC = max(p, pC)
    n_b = blocks.shape[1]

    Zsmooth, Vsmooth, VVsmooth, loglik = run_kf(y, A, C, Q, R, Z_0, V_0, debug_dir=debug_dir)

    A_new = A.copy()
    Q_new = Q.copy()
    V_0_new = V_0.copy()

    for i in range(n_b):
        r_i = r[i]
        rp = r_i * p
        rp1 = int(np.sum(r[:i]) * ppC)
        
        A_i = A[rp1:rp1 + r_i*ppC, rp1:rp1 + r_i*ppC]
        Q_i = Q[rp1:rp1 + r_i*ppC, rp1:rp1 + r_i*ppC]
        
        Z_slice_fwd = Zsmooth[rp1:rp1 + rp, 1:]
        Z_slice_bck = Zsmooth[rp1:rp1 + rp, :-1]
        V_slice_fwd = Vsmooth[rp1:rp1 + rp, rp1:rp1 + rp, 1:]
        V_slice_bck = Vsmooth[rp1:rp1 + rp, rp1:rp1 + rp, :-1]
        VV_slice = VVsmooth[rp1:rp1 + rp, rp1:rp1 + rp, :]
        
        EZZ = Z_slice_fwd @ Z_slice_fwd.T + np.sum(V_slice_fwd, axis=2)
        EZZ_BB = Z_slice_bck @ Z_slice_bck.T + np.sum(V_slice_bck, axis=2)
        EZZ_FB = Z_slice_fwd @ Z_slice_bck.T + np.sum(VV_slice, axis=2)

        A_i[:r_i, :rp] = EZZ_FB[:r_i, :rp] @ safe_inv(EZZ_BB[:rp, :rp])
        Q_i[:r_i, :r_i] = (EZZ[:r_i, :r_i] - A_i[:r_i, :rp] @ EZZ_FB[:r_i, :rp].T) / T
        
        A_new[rp1:rp1 + r_i*ppC, rp1:rp1 + r_i*ppC] = A_i
        Q_new[rp1:rp1 + r_i*ppC, rp1:rp1 + r_i*ppC] = Q_i
        V_0_new[rp1:rp1 + r_i*ppC, rp1:rp1 + r_i*ppC] = Vsmooth[rp1:rp1 + r_i*ppC, rp1:rp1 + r_i*ppC, 0]

    rp1 = int(np.sum(r) * ppC)
    niM = int(np.sum(i_idio[:nM]))

    if niM > 0:
        Z_idio_fwd = Zsmooth[rp1:, 1:]
        Z_idio_bck = Zsmooth[rp1:, :-1]
        
        EZZ = np.diag(np.diag(Z_idio_fwd @ Z_idio_fwd.T)) + np.diag(np.diag(np.sum(Vsmooth[rp1:, rp1:, 1:], axis=2)))
        EZZ_BB = np.diag(np.diag(Z_idio_bck @ Z_idio_bck.T)) + np.diag(np.diag(np.sum(Vsmooth[rp1:, rp1:, :-1], axis=2)))
        EZZ_FB = np.diag(np.diag(Z_idio_fwd @ Z_idio_bck.T)) + np.diag(np.diag(np.sum(VVsmooth[rp1:, rp1:, :], axis=2)))
        
        A_i = EZZ_FB @ np.diag(1.0 / np.diag(EZZ_BB))
        Q_i = (EZZ - A_i @ EZZ_FB.T) / T
        
        A_new[rp1:rp1+niM, rp1:rp1+niM] = A_i[:niM, :niM]
        Q_new[rp1:rp1+niM, rp1:rp1+niM] = Q_i[:niM, :niM]
        V_0_new[rp1:rp1+niM, rp1:rp1+niM] = np.diag(np.diag(Vsmooth[rp1:rp1+niM, rp1:rp1+niM, 0]))

    Z_0_new = Zsmooth[:, 0].reshape(-1, 1)
    
    nanY = np.isnan(y)
    y_copy = y.copy()
    y_copy[nanY] = 0

    C_new = C.copy()
    bl = np.unique(blocks, axis=0)
    n_bl = bl.shape[0]
    
    bl_idxM = []
    bl_idxQ = []
    R_con = np.zeros((0, 0))
    q_con = np.zeros((0, 1))
    
    for i in range(n_b):
        bl_idxQ.append(np.tile(bl[:, i:i+1], (1, r[i]*ppC)))
        tmp = np.hstack([np.tile(bl[:, i:i+1], (1, r[i])), np.zeros((n_bl, r[i]*(ppC-1)))])
        bl_idxM.append(tmp)
        R_con = la.block_diag(R_con, np.kron(R_mat, np.eye(r[i])))
        q_con = np.vstack([q_con, np.zeros((r[i]*R_mat.shape[0], 1))])
        
    bl_idxM = np.hstack(bl_idxM).astype(bool)
    bl_idxQ = np.hstack(bl_idxQ).astype(bool)

    m_states = Zsmooth.shape[0]
    bl_idxM_Z = np.zeros((n_bl, m_states), dtype=bool)
    bl_idxM_Z[:, :bl_idxM.shape[1]] = bl_idxM
    
    bl_idxQ_Z = np.zeros((n_bl, m_states), dtype=bool)
    bl_idxQ_Z[:, :bl_idxQ.shape[1]] = bl_idxQ

    if niM == 0:
        R_new = np.zeros((n, n))
        for t in range(T):
            nanYt = np.diag(~nanY[:, t])
            diff = y_copy[:, t:t+1] - nanYt @ C_new @ Zsmooth[:, t+1:t+2]
            R_new += diff @ diff.T + nanYt @ C_new @ Vsmooth[:, :, t+1] @ C_new.T @ nanYt + (np.eye(n) - nanYt) @ R @ (np.eye(n) - nanYt)
        R_new = np.diag(np.diag(R_new / T))
        
    else:
        i_idio_M = i_idio[:nM]
        n_idio_M = np.sum(i_idio_M)
        c_i_idio = np.cumsum(i_idio) - 1  
        
        for i in range(n_bl):
            bl_i = bl[i, :]
            rs = np.sum(r[bl_i])
            idx_i = np.where(np.all(blocks == bl_i, axis=1))[0]
            idx_iM = idx_i[idx_i < nM]
            n_i = len(idx_iM)
            
            denom = np.zeros((n_i*rs, n_i*rs))
            nom = np.zeros((n_i, rs))
            
            i_idio_i = i_idio_M[idx_iM]
            i_idio_ii = c_i_idio[idx_iM][i_idio_i]
            
            for t in range(T):
                nanYt = np.diag(~nanY[idx_iM, t])
                Z_b = Zsmooth[bl_idxM_Z[i, :], t+1:t+2]
                V_b = Vsmooth[np.ix_(bl_idxM_Z[i, :], bl_idxM_Z[i, :])][:, :, t+1]
                denom += np.kron(Z_b @ Z_b.T + V_b, nanYt)
                
                nom += y_copy[idx_iM, t:t+1] @ Z_b.T
                if np.any(i_idio_i):
                    Z_idio = Zsmooth[rp1 + i_idio_ii, t+1:t+2]
                    V_idio_b = Vsmooth[np.ix_(rp1 + i_idio_ii, bl_idxM_Z[i, :])][:, :, t+1]
                    nom -= nanYt[:, i_idio_i] @ (Z_idio @ Z_b.T + V_idio_b)
                    
            vec_C = safe_inv(denom) @ nom.flatten('F').reshape(-1, 1)
            C_new[np.ix_(idx_iM, bl_idxM_Z[i, :])] = vec_C.reshape((n_i, rs), order='F')
            
            idx_iQ = idx_i[idx_i >= nM]
            rps = rs * ppC
            
            R_con_i = R_con[:, bl_idxQ[i, :]]
            q_con_i = q_con.copy()
            no_c = ~np.any(R_con_i, axis=1)
            R_con_i = R_con_i[~no_c, :]
            q_con_i = q_con_i[~no_c, :]
            
            for j in idx_iQ:
                denom = np.zeros((rps, rps))
                nom = np.zeros((1, rps))
                idx_jQ = j - nM
                i_idio_jQ = np.arange(rp1 + n_idio_M + pC*idx_jQ, rp1 + n_idio_M + pC*(idx_jQ+1))
                
                V_0_new[np.ix_(i_idio_jQ, i_idio_jQ)] = Vsmooth[np.ix_(i_idio_jQ, i_idio_jQ, [0])][:,:,0]
                A_new[i_idio_jQ[0], i_idio_jQ[0]] = A_i[i_idio_jQ[0]-rp1, i_idio_jQ[0]-rp1]
                Q_new[i_idio_jQ[0], i_idio_jQ[0]] = Q_i[i_idio_jQ[0]-rp1, i_idio_jQ[0]-rp1]
                
                for t in range(T):
                    nanYt = 1.0 if not nanY[j, t] else 0.0
                    Z_b = Zsmooth[bl_idxQ_Z[i, :], t+1:t+2]
                    V_b = Vsmooth[np.ix_(bl_idxQ_Z[i, :], bl_idxQ_Z[i, :])][:, :, t+1]
                    denom += np.kron(Z_b @ Z_b.T + V_b, nanYt)
                    nom += y_copy[j, t] * Z_b.T
                    
                    R_vec = np.hstack([1, R_mat[:, 0]]).reshape(1, -1)
                    Z_idio = Zsmooth[i_idio_jQ, t+1:t+2]
                    V_idio_b = Vsmooth[np.ix_(i_idio_jQ, bl_idxQ_Z[i, :])][:, :, t+1]
                    nom -= nanYt * (R_vec @ Z_idio @ Z_b.T + R_vec @ V_idio_b)
                    
                C_i = safe_inv(denom) @ nom.T
                if R_con_i.size > 0:
                    inv_term = safe_inv(R_con_i @ safe_inv(denom) @ R_con_i.T)
                    C_i_constr = C_i - safe_inv(denom) @ R_con_i.T @ inv_term @ (R_con_i @ C_i - q_con_i)
                else:
                    C_i_constr = C_i
                C_new[j, bl_idxQ_Z[i, :]] = C_i_constr.flatten()

        R_new = np.zeros((n, n))
        for t in range(T):
            nanYt = np.diag(~nanY[:, t])
            diff = y_copy[:, t:t+1] - nanYt @ C_new @ Zsmooth[:, t+1:t+2]
            R_new += diff @ diff.T + nanYt @ C_new @ Vsmooth[:, :, t+1] @ C_new.T @ nanYt + (np.eye(n) - nanYt) @ R @ (np.eye(n) - nanYt)
            
        R_new = R_new / T
        RR = np.diag(R_new).copy()
        RR[:nM][i_idio_M] = 1e-04
        RR[nM:] = 1e-04
        R_new = np.diag(RR)

    # To match the legacy MATLAB/Octave implementation's behavior (which does not
    # update V_0 due to a bug where V_0 is returned instead of V_0_new), we return
    # the unmodified V_0.
    return C_new, R_new, A_new, Q_new, Z_0_new, V_0, loglik


def em_converged(loglik, previous_loglik, threshold=1e-4, check_increased=True):
    converged = False
    decrease = False
    if check_increased:
        if loglik - previous_loglik < -1e-3:
            decrease = True
    delta_loglik = abs(loglik - previous_loglik)
    avg_loglik = (abs(loglik) + abs(previous_loglik) + np.finfo(float).eps) / 2
    if (delta_loglik / avg_loglik) < threshold:
        converged = True
    return converged, decrease


def init_cond(x, r, p, blocks, optNaN, Rcon, q, NQ, i_idio, debug_dir=None):
    pC = Rcon.shape[1]
    ppC = max(p, pC)
    n_b = blocks.shape[1]
    
    xBal, indNaN = remnans_spline(x, optNaN, debug_dir=debug_dir)
    T, N = xBal.shape
    NM = N - NQ
    
    xNaN = xBal.copy()
    xNaN[indNaN] = np.nan
    C = np.zeros((N, 0))
    A = np.zeros((0, 0))
    Q = np.zeros((0, 0))
    initV = np.zeros((0, 0))
    
    res = xBal.copy()
    resNaN = xNaN.copy()
    indNaN[:pC-1, :] = True
    
    for i in range(n_b):
        r_i = r[i]
        C_i = np.zeros((N, r_i * ppC))
        idx_i = np.where(blocks[:, i])[0]
        idx_iM = idx_i[idx_i < NM]
        idx_iQ = idx_i[idx_i >= NM]
        
        cov_res = np.cov(res[:, idx_iM], rowvar=False, ddof=1)
        evals, evecs = la.eigh(cov_res)
        sort_idx = np.argsort(evals)[::-1]
        v = evecs[:, sort_idx[:r_i]]
        
        for j in range(v.shape[1]):
            non_zero_idx = np.where(np.abs(v[:, j]) > 1e-12)[0]
            if len(non_zero_idx) > 0:
                if v[non_zero_idx[0], j] < 0:
                    v[:, j] = -v[:, j]
                    
        C_i[idx_iM, :r_i] = v
        f = res[:, idx_iM] @ v
        F_cols = []
        for kk in range(max(p+1, pC)):
            start = pC - 1 - kk
            end = f.shape[0] - kk
            F_cols.append(f[start:end, :])
        F = np.hstack(F_cols)
        
        Rcon_i = np.kron(Rcon, np.eye(r_i))
        q_i = np.kron(q, np.zeros((r_i, 1)))
        ff = F[:, :r_i*pC]
        
        for j in idx_iQ:
            xx_j = resNaN[pC-1:, j]
            if np.sum(~np.isnan(xx_j)) < ff.shape[1] + 2:
                xx_j = res[pC-1:, j]
            ff_j = ff[~np.isnan(xx_j), :]
            xx_j = xx_j[~np.isnan(xx_j)].reshape(-1, 1)
            
            iff_j = safe_inv(ff_j.T @ ff_j)
            Cc = iff_j @ ff_j.T @ xx_j
            inv_Rcon = safe_inv(Rcon_i @ iff_j @ Rcon_i.T)
            Cc = Cc - iff_j @ Rcon_i.T @ inv_Rcon @ (Rcon_i @ Cc - q_i)
            C_i[j, :pC*r_i] = Cc.flatten()
            
        ff_full = np.vstack([np.zeros((pC-1, pC*r_i)), ff])
        res -= ff_full @ C_i.T
        resNaN = res.copy()
        resNaN[indNaN] = np.nan
        C = np.hstack([C, C_i])

        z = F[:, :r_i]
        Z_mat = F[:, r_i : r_i*(p+1)]
        A_i = np.zeros((r_i*ppC, r_i*ppC))
        A_temp = safe_inv(Z_mat.T @ Z_mat) @ Z_mat.T @ z
        A_i[:r_i, :r_i*p] = A_temp.T
        A_i[r_i:, :r_i*(ppC-1)] = np.eye(r_i*(ppC-1))

        Q_i = np.zeros((ppC*r_i, ppC*r_i))
        e_resid = z - Z_mat @ A_temp
        Q_i[:r_i, :r_i] = np.cov(e_resid, rowvar=False, ddof=1)
        
        kron_A = np.kron(A_i, A_i)
        initV_i = safe_inv(np.eye((r_i*ppC)**2) - kron_A) @ Q_i.flatten('F')
        initV_i = initV_i.reshape((r_i*ppC, r_i*ppC), order='F')

        A = la.block_diag(A, A_i) if A.size else A_i
        Q = la.block_diag(Q, Q_i) if Q.size else Q_i
        initV = la.block_diag(initV, initV_i) if initV.size else initV_i

    R = np.diag(np.nanvar(resNaN, axis=0, ddof=1))
    eyeN = np.eye(N)

    ii_idio = np.where(i_idio)[0]
    n_idio = len(ii_idio)
    B = np.zeros((n_idio, n_idio))
    S = np.zeros((n_idio, n_idio))

    if n_idio == 0:
        R = np.diag(np.nanvar(resNaN, axis=0, ddof=1))
        initZ = np.zeros((A.shape[0], 1))
    else:
        C = np.hstack([C, eyeN])
        BM = np.zeros((n_idio, n_idio))
        SM = np.zeros((n_idio, n_idio))
        
        for i in range(n_idio):
            R[ii_idio[i], ii_idio[i]] = 1e-04
            res_i = resNaN[:, ii_idio[i]]
            valid_idx = np.where(~np.isnan(res_i))[0]
            leadZero = valid_idx.min() if len(valid_idx) > 0 else 0
            res_i_clean = res[leadZero:, ii_idio[i]].reshape(-1, 1)
            
            if len(res_i_clean) < 3:
                BM[i, i] = 0.0
                SM[i, i] = 1e-5
            else:
                denom = res_i_clean[:-1].T @ res_i_clean[:-1]
                if np.abs(denom) < 1e-12:
                    ar_coef_val = 0.0
                else:
                    ar_coef_val = float((safe_inv(denom) @ res_i_clean[:-1].T @ res_i_clean[1:]).item())
                if np.isnan(ar_coef_val):
                    ar_coef_val = 0.0
                BM[i, i] = float(np.clip(ar_coef_val, -0.99, 0.99))
                
                resid_SM = res_i_clean[1:]
                cov_val = float(np.cov(resid_SM.flatten(), ddof=1)) if len(resid_SM) > 1 else 1e-5
                if np.isnan(cov_val) or cov_val < 1e-6:
                    cov_val = 1e-5
                SM[i, i] = cov_val
            
        bm_diag = np.diag(BM)
        denom_bm = np.maximum(1.0 - bm_diag**2, 1e-4)
        initViM = np.diag(1.0 / denom_bm) * SM
        
        temp_C_append = np.vstack([np.zeros((NM, pC*NQ)), np.kron(np.eye(NQ), np.hstack([1, Rcon[:, 0]]))] )
        C = np.hstack([C, temp_C_append])
        
        Rdiag = np.diag(R).copy()
        sig_e = Rdiag[NM:] / 19.0
        Rdiag[:NM][i_idio[:NM]] = 1e-04  
        Rdiag[NM:] = 1e-04
        R = np.diag(Rdiag)
        
        rho0 = 0.1
        block_BQ = np.vstack([[rho0] + [0]*(pC-1), np.hstack([np.eye(pC-1), np.zeros((pC-1, 1))])])
        BQ = np.kron(np.eye(NQ), block_BQ)
        
        temp_SQ = np.zeros((pC, pC))
        temp_SQ[0, 0] = 1.0
        SQ = np.kron(np.diag((1 - rho0**2) * sig_e), temp_SQ)
        
        kron_BQ = np.kron(BQ, BQ)
        initViQ = safe_inv(np.eye((pC*NQ)**2) - kron_BQ) @ SQ.flatten('F')
        initViQ = initViQ.reshape((pC*NQ, pC*NQ), order='F')
        
        A = la.block_diag(A, BM, BQ)
        Q = la.block_diag(Q, SM, SQ)
        initZ = np.zeros((A.shape[0], 1))
        initV = la.block_diag(initV, initViM, initViQ)

    return A, C, Q, R, initZ, initV


# Legacy alias
em_dfm_ss_block_idioqarma_restrmq = fit_dfm_em


def winsorize_data_internal(X, k=4.0):
    X_wins = X.copy()
    T, N = X.shape
    for j in range(N):
        col = X[:, j]
        med = np.nanmedian(col)
        mad = np.nanmedian(np.abs(col - med))
        if mad < 1e-6:
            mad = 1.0
        lower_bound = med - k * mad
        upper_bound = med + k * mad
        non_nan_mask = ~np.isnan(col)
        X_wins[non_nan_mask, j] = np.clip(col[non_nan_mask], lower_bound, upper_bound)
    return X_wins


@njit
def numba_pinv(A):
    U, S, Vt = np.linalg.svd(A)
    S_inv = np.zeros(len(S))
    for i in range(len(S)):
        if S[i] > 1e-12:
            S_inv[i] = 1.0 / S[i]
    return Vt.T @ np.diag(S_inv) @ U.T


@njit
def _carter_kohn_draw_jit(y, A, C, Q, R_diag, x_0, Sig_0):
    n, nobs = y.shape
    m = A.shape[0]
    
    am = np.zeros((m, nobs))
    Pm = np.zeros((m, m, nobs))
    amU = np.zeros((m, nobs + 1))
    PmU = np.zeros((m, m, nobs + 1))
    
    Au = x_0.copy().reshape(-1, 1)
    Pu = Sig_0.copy()
    amU[:, 0] = Au.flatten()
    PmU[:, :, 0] = Pu
    
    for t in range(nobs):
        A_pred = A @ Au
        P_pred = A @ Pu @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        y_t = y[:, t:t+1]
        
        # Gather valid non-NaN indices
        ix_list = []
        for i in range(n):
            if not np.isnan(y_t[i, 0]):
                ix_list.append(i)
                
        if len(ix_list) == 0:
            Au = A_pred
            Pu = P_pred
        else:
            y_obs = np.zeros((len(ix_list), 1))
            C_obs = np.zeros((len(ix_list), m))
            R_obs = np.zeros((len(ix_list), len(ix_list)))
            for idx, i in enumerate(ix_list):
                y_obs[idx, 0] = y_t[i, 0]
                C_obs[idx, :] = C[i, :]
                R_obs[idx, idx] = R_diag[i]
                
            PZ = P_pred @ C_obs.T
            F_inv = np.linalg.inv(C_obs @ PZ + R_obs)
            PZF = PZ @ F_inv
            V = y_obs - C_obs @ A_pred
            Au = A_pred + PZF @ V
            Pu = P_pred - PZF @ PZ.T
            Pu = 0.5 * (Pu + Pu.T)
            
        am[:, t] = A_pred.flatten()
        Pm[:, :, t] = P_pred
        amU[:, t + 1] = Au.flatten()
        PmU[:, :, t + 1] = Pu

    states_draw = np.zeros((m, nobs + 1))
    cov_T = PmU[:, :, nobs]
    mean_T = amU[:, nobs]
    U, S, Vt = np.linalg.svd(cov_T)
    S_sqrt = np.sqrt(np.where(S > 1e-10, S, 0.0))
    states_draw[:, nobs] = mean_T + U @ (S_sqrt * np.random.normal(0.0, 1.0, m))
    
    for t in range(nobs - 1, -1, -1):
        Pu_t = PmU[:, :, t]
        P_pred_next = Pm[:, :, t]
        mean_u_t = amU[:, t]
        
        J_t = Pu_t @ A.T @ numba_pinv(P_pred_next)
        mean_cond = mean_u_t + J_t @ (states_draw[:, t + 1] - am[:, t])
        cov_cond = Pu_t - J_t @ A @ Pu_t
        cov_cond = 0.5 * (cov_cond + cov_cond.T)
        
        U, S, Vt = np.linalg.svd(cov_cond)
        S_sqrt = np.sqrt(np.where(S > 1e-10, S, 0.0))
        states_draw[:, t] = mean_cond + U @ (S_sqrt * np.random.normal(0.0, 1.0, m))
        
    return states_draw


def carter_kohn_draw(y, A, C, Q, R_diag, x_0, Sig_0):
    if _HAS_NUMBA:
        return _carter_kohn_draw_jit(y, A, C, Q, R_diag, x_0, Sig_0)
        
    n, nobs = y.shape
    m = A.shape[0]
    
    am = np.zeros((m, nobs))
    Pm = np.zeros((m, m, nobs))
    amU = np.zeros((m, nobs + 1))
    PmU = np.zeros((m, m, nobs + 1))
    
    Au = x_0.copy().reshape(-1, 1)
    Pu = Sig_0.copy()
    amU[:, 0] = Au.flatten()
    PmU[:, :, 0] = Pu
    
    for t in range(nobs):
        A_pred = A @ Au
        P_pred = A @ Pu @ A.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        y_t = y[:, t:t+1]
        ix = ~np.isnan(y_t).flatten()
        
        if not np.any(ix):
            Au = A_pred
            Pu = P_pred
        else:
            y_obs = y_t[ix]
            C_obs = C[ix, :]
            R_obs = np.diag(R_diag[ix])
            
            PZ = P_pred @ C_obs.T
            F_inv = np.linalg.inv(C_obs @ PZ + R_obs)
            PZF = PZ @ F_inv
            V = y_obs - C_obs @ A_pred
            Au = A_pred + PZF @ V
            Pu = P_pred - PZF @ PZ.T
            Pu = 0.5 * (Pu + Pu.T)
            
        am[:, t] = A_pred.flatten()
        Pm[:, :, t] = P_pred
        amU[:, t + 1] = Au.flatten()
        PmU[:, :, t + 1] = Pu

    states_draw = np.zeros((m, nobs + 1))
    cov_T = PmU[:, :, nobs]
    mean_T = amU[:, nobs]
    U, S, V = np.linalg.svd(cov_T)
    S_sqrt = np.sqrt(np.where(S > 1e-10, S, 0.0))
    states_draw[:, nobs] = mean_T + U @ (S_sqrt * np.random.normal(size=m))
    
    for t in range(nobs - 1, -1, -1):
        Pu_t = PmU[:, :, t]
        P_pred_next = Pm[:, :, t]
        mean_u_t = amU[:, t]
        
        J_t = Pu_t @ A.T @ np.linalg.pinv(P_pred_next)
        mean_cond = mean_u_t + J_t @ (states_draw[:, t + 1] - am[:, t])
        cov_cond = Pu_t - J_t @ A @ Pu_t
        cov_cond = 0.5 * (cov_cond + cov_cond.T)
        
        U, S, V = np.linalg.svd(cov_cond)
        S_sqrt = np.sqrt(np.where(S > 1e-10, S, 0.0))
        states_draw[:, t] = mean_cond + U @ (S_sqrt * np.random.normal(size=m))
        
    return states_draw


def fit_dfm_mcmc(X, Par, Res_old=None):
    from scipy.stats import invgamma
    
    from dfm_nowcast.data import DFMPreprocessor
    preprocessor = DFMPreprocessor(
        winsorization=Par.get('winsorization', False),
        winsorization_k=Par.get('winsorization_k', 4.0),
        robust_scaling=Par.get('robust_scaling', False)
    )
    if Res_old is None or not Res_old:
        X_train = preprocessor.fit_transform(X)
        Mx = preprocessor.Mx
        Wx = preprocessor.Wx
    else:
        Mx = Res_old['Mx']
        Wx = Res_old['Wx']
        preprocessor.Mx = Mx
        preprocessor.Wx = Wx
        X_train = preprocessor.transform(X)
        
    T_len, N = X.shape
    
    r = int(Par['r'][0])
    p = int(Par['p'])
    n_draws = int(Par.get('mcmc_draws', 100))
    burn_in = int(Par.get('mcmc_burnin', 50))
    
    y = X_train.T
    m = r * p
    
    # Warm-starting / Initialization
    if (Res_old is not None and isinstance(Res_old, dict) and 'C' in Res_old 
            and Res_old['C'].shape == (N, m) and Res_old['A'].shape == (m, m)):
        C = Res_old['C'].copy()
        R_diag = np.diag(Res_old['R']).copy()
        A = Res_old['A'].copy()
        Q = Res_old['Q'].copy()
        x_0 = Res_old['Z_0'].copy() if 'Z_0' in Res_old else np.zeros(m)
        Sig_0 = Res_old['V_0'].copy() if 'V_0' in Res_old else np.eye(m)
    else:
        C = np.zeros((N, m))
        C[:, 0] = np.random.normal(0, 0.1, N)
        R_diag = np.ones(N) * 0.5
        A = np.zeros((m, m))
        A[0:r, 0:r] = np.eye(r) * 0.8
        if p > 1:
            A[r:m, 0:r*(p-1)] = np.eye(r*(p-1))
        Q = np.eye(m) * 0.1
        Q[r:m, r:m] = 0.0
        
        x_0 = np.zeros(m)
        Sig_0 = np.eye(m)
    
    C_draws = []
    R_draws = []
    A_draws = []
    Q_draws = []
    factor_draws = []
    
    sigma_c2_prior = 1.0
    a0 = 2.01
    b0 = 0.5
    
    for draw in range(n_draws):
        states = carter_kohn_draw(y, A, C, Q, R_diag, x_0, Sig_0)
        factors = states[:, 1:]
        
        # Draw C and R
        for i in range(N):
            y_i = y[i, :]
            ix = ~np.isnan(y_i)
            y_obs = y_i[ix]
            F_obs = factors[:, ix].T
            
            V_prior_inv = (1.0 / sigma_c2_prior) * np.eye(m)
            V_post = np.linalg.inv(V_prior_inv + (1.0 / R_diag[i]) * F_obs.T @ F_obs)
            mu_post = V_post @ ((1.0 / R_diag[i]) * F_obs.T @ y_obs)
            C[i, :] = np.random.multivariate_normal(mu_post, V_post)
            
            residuals = y_obs - F_obs @ C[i, :]
            a_post = a0 + len(y_obs) / 2.0
            b_post = b0 + np.sum(residuals**2) / 2.0
            R_diag[i] = max(invgamma.rvs(a_post, scale=b_post), 1e-4)
            
        # Draw VAR transition A and Q
        if p == 2:
            F_t = factors[0:r, 2:]
            F_lag = np.vstack((factors[0:r, 1:-1], factors[0:r, 0:-2]))
            
            lambda_var = 0.1
            V_A_inv = lambda_var * np.eye(2*r)
            V_A_post = np.linalg.inv(V_A_inv + F_lag @ F_lag.T)
            mu_A_post = V_A_post @ (F_lag @ F_t.T)
            
            A_row = np.zeros((r, 2*r))
            for j in range(r):
                cov_j = Q[j, j] * V_A_post
                A_row[j, :] = np.random.multivariate_normal(mu_A_post[:, j], cov_j)
                
            A[0:r, :] = A_row
            
            u = F_t - A_row @ F_lag
            for j in range(r):
                a_q = a0 + u.shape[1] / 2.0
                b_q = b0 + np.sum(u[j, :]**2) / 2.0
                Q[j, j] = max(invgamma.rvs(a_q, scale=b_q), 1e-4)
                
        if draw >= burn_in:
            C_draws.append(C.copy())
            R_draws.append(R_diag.copy())
            A_draws.append(A.copy())
            Q_draws.append(Q.copy())
            factor_draws.append(factors.copy())
            
    Res = {
        'C': np.mean(C_draws, axis=0),
        'R': np.diag(np.mean(R_draws, axis=0)),
        'A': np.mean(A_draws, axis=0),
        'Q': np.mean(Q_draws, axis=0),
        'F': np.mean(factor_draws, axis=0).T,
        'Z_0': x_0,
        'V_0': Sig_0,
        'Mx': Mx,
        'Wx': Wx,
        'r': Par['r'],
        'p': Par['p']
    }
    
    if Par.get('covariance_regularization', False):
        lambda_val = Par.get('ridge_lambda', 1e-4)
        Res['R'] += lambda_val * np.eye(Res['R'].shape[0])
        
    x_sm = Res['F'] @ Res['C'].T
    Res['X_sm'] = preprocessor.inverse_transform(x_sm)
    return Res


# Legacy alias
fit_dfm_mcmc_internal = fit_dfm_mcmc
carter_kohn_draw_internal = carter_kohn_draw



