import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from scipy.interpolate import interp1d


def ExtractPath_LARS(x, y, which_Cov, betaNULL = 0, multiTest = False):
    '''
    Extract the path information from Lars
    Parameters:
        x: numpy array
        y: numpy vector
        which_Cov: int or a list
        betaNULL: int or a list
        multiTest: boolean
    Return:
        a tuple of lambda, beta matrix, beta matrix after removal
    
    '''    
    n, p = x.shape
      

    if multiTest:
        
        adj_Y = y - np.sum(betaNULL * x[:, which_Cov], axis = 1)
        X_scaled = preprocessing.scale(x)
        
        alphas, _, coefs  = linear_model.lars_path(X_scaled, adj_Y, Gram = None, 
                                                   method = 'lasso', return_path = True)
        alphas_j, _, coefs_j  = linear_model.lars_path(np.delete(X_scaled, which_Cov, axis = 1), adj_Y, Gram=None, 
                                                       method = 'lasso', return_path=True)
        
        union_lambda = np.sort(np.unique(np.append(alphas, alphas_j)))
        
        
        coefs_interp = np.apply_along_axis(lambda x: 
                         interp1d(alphas, x, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs)
        coefs_interp_j = np.apply_along_axis(lambda x: 
                         interp1d(alphas_j, x, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs_j)
        
        coefs_interp_j = np.insert(coefs_interp_j, which_Cov, 0, axis = 0)
        
    elif not multiTest:
        
        adj_Y = y - betaNULL * x[:, which_Cov]
        X_scaled = preprocessing.scale(x)
        
        alphas, _, coefs  = linear_model.lars_path(X_scaled, adj_Y, Gram = None, 
                                                   method = 'lasso', return_path = True)
        alphas_j, _, coefs_j  = linear_model.lars_path(np.delete(X_scaled, which_Cov, axis = 1), adj_Y, Gram=None, 
                                                       method = 'lasso', return_path=True)
        
        union_lambda = np.sort(np.unique(np.append(alphas, alphas_j)))
        
        
        coefs_interp = np.apply_along_axis(lambda t: 
                         interp1d(alphas, t, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs)
        coefs_interp_j = np.apply_along_axis(lambda t: 
                         interp1d(alphas_j, t, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs_j)
        
        coefs_interp_j = np.insert(coefs_interp_j, which_Cov, 0, axis = 0)
        
        
        
    return union_lambda, coefs_interp, coefs_interp_j


def ExtractPath_ENET(x, y, which_Cov, betaNULL = 0, multiTest = False, l1_ratio=1):
    '''
    Extract the path information from Lars
    Parameters:
        x: numpy array
        y: numpy vector
        which_Cov: int or a list
        betaNULL: int or a list
        multiTest: boolean
    Return:
        a tuple of lambda, beta matrix, beta matrix after removal
    
    '''    
    n, p = x.shape
      

    if multiTest:
        
        adj_Y = y - np.sum(betaNULL * x[:, which_Cov], axis = 1)
        X_scaled = preprocessing.scale(x)
        
        alphas, coefs, _ = linear_model.enet_path(X_scaled, adj_Y, l1_ratio=l1_ratio, eps=0.001, n_alphas=100, 
                                            alphas=None, precompute=False, fit_intercept=False)
        alphas_j, coefs_j, _  = linear_model.enet_path(np.delete(X_scaled, which_Cov, axis = 1), adj_Y, l1_ratio=l1_ratio, eps=0.001, n_alphas=100, 
                                            alphas=None, precompute=False, fit_intercept=False)
        
        union_lambda = np.sort(np.unique(np.append(alphas, alphas_j)))
        
        
        coefs_interp = np.apply_along_axis(lambda t: 
                         interp1d(alphas, t, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs)
        coefs_interp_j = np.apply_along_axis(lambda t: 
                         interp1d(alphas_j, t, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs_j)
        
        coefs_interp_j = np.insert(coefs_interp_j, which_Cov, 0, axis = 0)
        
    elif not multiTest:
        
        adj_Y = y - betaNULL * x[:, which_Cov]
        X_scaled = preprocessing.scale(x)
        
        alphas, coefs, _ = linear_model.enet_path(X_scaled, adj_Y, l1_ratio=l1_ratio, eps=0.001, n_alphas=100, 
                                            alphas=None, precompute=False, fit_intercept=False)
        alphas_j, coefs_j, _  = linear_model.enet_path(np.delete(X_scaled, which_Cov, axis = 1), adj_Y, l1_ratio=l1_ratio, eps=0.001, n_alphas=100, 
                                            alphas=None, precompute=False, fit_intercept=False)
        
        union_lambda = np.sort(np.unique(np.append(alphas, alphas_j)))
        
        
        coefs_interp = np.apply_along_axis(lambda t: 
                         interp1d(alphas, t, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs)
        coefs_interp_j = np.apply_along_axis(lambda t: 
                         interp1d(alphas_j, t, kind='linear', bounds_error=False, fill_value=0)(union_lambda),
                         1, coefs_j)
        
        coefs_interp_j = np.insert(coefs_interp_j, which_Cov, 0, axis = 0)
        
        
        
    return union_lambda, coefs_interp, coefs_interp_j



def LOCO_TS(obj):
    
    union_lambda, coefs_interp, coefs_interp_j = obj
    
    M = len(union_lambda)
    Delta = coefs_interp_j.transpose() - coefs_interp.transpose()
    Delta_1 = np.delete(Delta, M-1, axis = 0)
    Delta_2 = np.delete(Delta, 1, axis =0)
    Lambda = np.diff(union_lambda)
    Epsilon = 1/3 * Lambda[:, np.newaxis] * (Delta_1 * Delta_1 + Delta_1 * Delta_2 + Delta_2 * Delta_2)

    return Epsilon.sum()