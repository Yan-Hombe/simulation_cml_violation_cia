##############################################################################
# DGP, simulation, and DGP vizualization
# this code is used to generate the data for the simulation study in the paper
# output: csv file with all results of the simulation
##############################################################################

import numpy as np
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import time
from sklearn.preprocessing import StandardScaler
import warnings

import scienceplots  # Importing this enables the science style

# statistical models packages
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from doubleml import DoubleMLIRM, DoubleMLPLR, DoubleMLData
from xgboost import XGBRegressor, XGBClassifier
from xbcausalforest import XBCF

# meta learner packages
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseRRegressor, BaseXRegressor, TMLELearner

# own functions
from maincode import algorithms as algo # machine learning algos for ate
# check current wd
import os
# change wd
os.chdir('Master_Thesis')
os.getcwd()


# new beta
# create betas 
def beta(n_conf): 


    k = np.array(range(n_conf)) + 1

    # create beta array with betas and probabilities
    beta_array = 1 - k/20

    return beta_array

# DGP complex, non linear artificial data
def generate_X(n_observations, n_conf):

    # generat each X from a normal distribution with mean 0 and variance 1
    x_matrix = np.random.normal(0, 1, size=(n_observations, n_conf))

    return x_matrix

X = generate_X(n_observations=10, n_conf=2)

# di generator with Mareckovas approach 
def D_i_generator(x_matrix, linear_di=True, linear=False, lam = 1):
    n_conf = x_matrix.shape[1]
    u_i = np.random.normal(0, 1, len(x_matrix)) # always N(0,1)

    if linear_di and not linear:
        b_k = np.full(n_conf, 1)
        d_i = lam * np.dot(x_matrix, b_k) + u_i
        fx_ij = d_i - u_i

    elif not linear_di and not linear:
        trans = non_lin_trans(x_matrix)
        # Adjust scaling factor if necessary
        d_i = lam * (trans / np.std(trans)) + u_i
        fx_ij = d_i - u_i

    elif linear:
        d_i = lam * (x_matrix @ beta(n_conf)) + u_i
        fx_ij = d_i - u_i

    median = np.median(d_i)
    # Convert d_i to binary based on the median
    d_i = np.where(d_i > (median), 1, 0)


    cdf_value = fx_ij - median

    probabilities = norm.cdf(cdf_value)

    # Clip probabilities to avoid extreme values
    #probabilities = np.clip(probabilities, 0.05, 0.95)

    return d_i, probabilities

def non_lin_trans(X):
    n_conf = len(X[0])
    ns = (n_conf)
    c = 1
    k = np.arange(1, ns+1)
    r = c - (c/ns)*k
    X_abs = np.abs(X - r)
    X_abs = np.sum(X_abs, axis=1)
    X_abs = X_abs 

    x_end = n_conf-1
    x_mid = int(round((n_conf/2),0))-1

    X_inter1= X[:,0]* X[:,x_mid] * (X[:, x_end])

    X_inter2 = (((X[:,1] -3)* (X[:, x_mid] + 3))**2)

    X_inter2 = X_inter2/  np.mean(X_inter2)

    X_inter3 = np.log(np.abs(X[:,2])) * (X[:, (x_end-1)] - 3) 

    combined = X_abs + X_inter1 + X_inter2 + X_inter3
    return combined

# create variable of confounders, mu(x)
def pi_i_generator(x_matrix, linear_pi = None, linear = None):
    n_conf = len(x_matrix[0])
    v_i = np.random.normal(0, 1, len(x_matrix))

    if linear:
        pi_i = x_matrix @ beta(n_conf)# + v_i

    elif not linear_pi:

        pi_i = non_lin_trans(x_matrix)# + v_i

    return pi_i, v_i


# function for th DGP
def DGP(x_matrix, B_T, gamma, linear_pi, linear_di, linear, lam):
    d_i, probabilities = D_i_generator(x_matrix,linear_di=linear_di, linear=linear, lam = lam) # get d_i the treatment variable
    pi_i, v_i = pi_i_generator(x_matrix, linear_pi = linear_pi, linear= linear) # get mu_i the confounder

    y_i = B_T * d_i + gamma * pi_i + v_i
    return y_i, d_i, v_i

# test DGP for real world data
def not_run():
    B_T = 1
    gamma = 1
    n_conf = 5
    n_observations = 100
    x_matrix = real_world_matrix
    Data = DGP(x_matrix = x_matrix, B_T = B_T, gamma = gamma, linear_pi=False, linear_di=True, scarcity=False)
    print(Data)



# NEW VERSION OF FUNCTION
# function that omits confounders in underlying data 
def confounder_omiting(x_matrix, share_omit:float, random_omit = False):
    conf_omit =  np.round((len(x_matrix[0]) * share_omit), 0).astype(int)
    if conf_omit == len(x_matrix[0]):
        conf_omit = len(x_matrix[0]) - 1
    
    # create randomly omitting confounders
    if random_omit == True: 
        omit = random.sample(range(0, len(x_matrix[0])), conf_omit)
        x_matrix_omit = np.delete(x_matrix, omit, axis=1)
    elif random_omit == False:
        x_matrix_omit = x_matrix[:, :(len(x_matrix[0])-conf_omit)]
    return x_matrix_omit

def translate_to_np_r(strings):
    result = []
    for s in strings:
        # Translate each string to np.r_ compatible format
        parts = []
        # Split the string into individual components (variables or ranges)
        tokens = s.split()
        
        for token in tokens:
            # If it's a range (e.g., X9-X18), convert it to slice format
            if '-' in token:
                start, end = map(lambda x: int(x[1:]), token.split('-'))  # extract numbers from X9-X18
                parts.append(f"{start}:{end + 1}")  # add 1 because end index is exclusive
            else:
                # If it's a single variable (e.g., X1), just extract the number
                parts.append(f"{int(token[1:])}")
        
        # Join all parts for np.r_ use
        result.append(f"np.r_[{', '.join(parts)}]")  # e.g., np.r_[1, 2, 9:19]
    
    return result

# VERSION FOR SCENARIOS 
# function that omits confounders in underlying data 
def confounder_omiting(x_matrix, share_omit:str, random_omit = False):
    
    slicer = eval(share_omit)
    if slicer[0] == 999:
        x_matrix_omit = x_matrix
    else:
        slicer_0_based = slicer-1
        x_matrix_omit = np.delete(x_matrix, slicer_0_based, axis=1)
    
    return x_matrix_omit


# test confounder_omiting
def not_run():
    x_matrix = generate_X(n_observations=2, n_conf=20)
    share_omit_string = ['X999', 'X1', 'X2', 'X15', 'X16', 'X1 X16', 'X1 X2', 'X15 X16', 'X10-X19', 'X1-X5 X15-X19', 'X1-X10', 'X3-X20']
    strengthScenario = ['no_omit', 'S', 'S', 'W', 'W', 'S W', 'S S', 'W W', '10W', '5S 5W', '10S','all but 2 strong']
    share_omit = translate_to_np_r(share_omit_string)
    share_omit = share_omit[-1]
    x_matrix_omit = confounder_omiting(x_matrix, share_omit=share_omit, random_omit=False)
    print(x_matrix_omit)
## ----------------------

def bias_matrix(iterations_monte_carlo, m_ML):
    """
    create empty n x m matrix for the results

    Parameters
    ----------
    m_ML: TYPE: int
        DESCRIPTION: number of machine learning algorithms
    iterations_monte_carlo: TYPE: int
        DESCRIPTION: number of iterations for the monte carlo simulation
    ----------
    """
    # Define the shape of the matrix
    shape = (iterations_monte_carlo, m_ML) 
    # Create a matrix filled with NaN values
    results_ar = np.full(shape, np.nan)

    return results_ar

def time_matrix(iterations_monte_carlo, m_ML):
    """
    create empty n x m matrix for the results

    Parameters
    ----------
    m_ML: TYPE: int
        DESCRIPTION: number of machine learning algorithms
    iterations_monte_carlo: TYPE: int
        DESCRIPTION: number of iterations for the monte carlo simulation
    ----------
    """
    if iterations_monte_carlo < 10:
        n_time_rows = 0
        
    else:
        n_time_rows = int((iterations_monte_carlo - (iterations_monte_carlo % 10)) / 10)

    # Define the shape of the matrix
    shape = (n_time_rows, m_ML) 
    # Create a matrix filled with NaN values
    results_ar = np.full(shape, np.nan)

    return results_ar

# create function to combine all ATE algorithm functions into one generic function
def ATE_hat_generic(treat, outcome, exog, current_ml_algo, propensity_ML = None, outcome_ML = None, binary_treatment = None, n_folds = None):
    
    method = getattr(algo, current_ml_algo)

    ATE_hat = method(treat, outcome, exog, propensity_ML = propensity_ML, 
                outcome_ML = outcome_ML, binary_treatment = binary_treatment, n_folds = n_folds)
    
    return ATE_hat

# monte carlo simulation
def monte_carlo_simulation(n_conf, n_observations, iterations_monte_carlo, m_ML, 
                           share_omit, B_T, gamma, lam, linear_pi, linear_di, linear,
                           random_omit, real_world_matrix = None,
                           outcome_nuisance = XGBRegressor(), propensity_nuisance = RandomForestClassifier()):

    # create empty bias matrix
    bias_mat = bias_matrix(iterations_monte_carlo = iterations_monte_carlo, m_ML=m_ML)

    # create empty time matrix
    time_mat = time_matrix(iterations_monte_carlo = iterations_monte_carlo, m_ML=m_ML)

    time_counter = 1 # counter for knowing when to evaluate time

    # create list of all machine learning algorithms by the function names
    all_algorithms = ['lin_reg', 'aipw_cv', 'meta_t_learner', 'meta_x_learner', 'XBCF_ate']

    if len(all_algorithms) != m_ML:
        raise ValueError('Number of machine learning algorithms (m_ML) in function monte_carlo_simulation does not match the number of algorithms in the list of all_algorithms')

    # start outer loop DGP
    for iteration in range(iterations_monte_carlo):
    


        x_matrix = generate_X(n_observations=n_observations, n_conf=n_conf)

        Data = DGP(x_matrix = x_matrix, B_T = B_T, gamma = gamma, lam=lam, linear=linear, linear_pi=linear_pi, linear_di=linear_di)

        outcome = Data[0]
        treat = Data[1].astype(np.int32)

        # omit confounders
        exog = confounder_omiting(x_matrix = x_matrix, share_omit = share_omit, random_omit=random_omit)

        take_time = False
        if time_counter % 10 == 0:
            take_time = True
        
        folds_aipw = 3 # specify the folds of cross fitting for DML
        algo_counter = 0 # counter for filling the bias matrix

        for algorithm in all_algorithms:

            if take_time:
                print('---------')
                print('---------')
                print('---------')
                print('take time', time_counter)
                start = time.time()
                # get ATE for all algorithms
                try: 
                    ATE_hat = ATE_hat_generic(treat, outcome, exog, algorithm,
                                            propensity_ML = propensity_nuisance, outcome_ML = outcome_nuisance, 
                                            binary_treatment = True, n_folds = folds_aipw)

                    bias_mat[iteration, algo_counter] = (ATE_hat - B_T).item()
            
                    #print(ATE_hat, algorithm)
                except:
                    print('the following algorithm failed:', algorithm)
                end = time.time()
                
                # devide the time_counter by 10 to get the right position in the time matrix and subtract 1 due to staring at 1
                time_mat_position = int((time_counter / 10) - 1)

                time_mat[time_mat_position, algo_counter] = end - start

            else:
                try: 
                    ATE_hat = ATE_hat_generic(treat, outcome, exog, algorithm,
                                            propensity_ML = propensity_nuisance, outcome_ML = outcome_nuisance, 
                                            binary_treatment = True, n_folds = folds_aipw)

                    bias_mat[iteration, algo_counter] = (ATE_hat - B_T).item()
            
                    print(ATE_hat, algorithm)
                except:
                    print('the following algorithm failed:', algorithm)

            algo_counter += 1 # increase the counter for inserting the bias into the bias matrix
        
        time_counter += 1
        take_time = False

    return bias_mat, time_mat

# calculate bias, mse etc. from bias matrix
def bias_time_aggregation(bias_mat, B_T, time_mat):
    """
    calculate the bias, MSE, percentage and empirical SE and the monte carlo SE of these measures

    Parameters
    ----------
    bias_mat : TYPE: np.array
        DESCRIPTION: matrix of bias
    B_T : TYPE: float
        DESCRIPTION: true treatment effect
    time_mat : TYPE: np.array
        DESCRIPTION: matrix of time
        
    Returns
    -------
    bias : TYPE: np.array
        DESCRIPTION: bias
    Monte_carlo_SE_bias : TYPE: np.array
        DESCRIPTION: monte carlo SE of the bias
    prct_bias : TYPE: np.array
        DESCRIPTION: percentage bias
    Monte_carlo_SE_prct_bias : TYPE: np.array
        DESCRIPTION: monte carlo SE of the percentage bias
    MSE : TYPE: np.array
        DESCRIPTION: MSE
    Monte_carlo_SE_MSE : TYPE: np.array
        DESCRIPTION: monte carlo SE of the MSE
    empirical_SE : TYPE: np.array
        DESCRIPTION: empirical SE
    Monte_carlo_SE_empirical : TYPE: np.array
        DESCRIPTION: monte carlo SE of the empirical SE
    time : TYPE: np.array  
    """
    # count the number of nan in the bias matrix
    bool_mat = np.isnan(bias_mat)
    nancount = np.count_nonzero(bool_mat, axis=0)

    # calculate bias and monte carlo SE of the bias
    bias  = np.nanmean(bias_mat, axis=0)
    n_sim = bias_mat.shape[0]
    Monte_carlo_SE_bias = np.nanstd(bias_mat, axis=0) / np.sqrt(n_sim - 1)

    prct_bias = (bias / B_T) * 100
    Monte_carlo_SE_prct_bias = np.nanstd((bias_mat / B_T)*100, axis=0) / np.sqrt(n_sim - 1)

    # absolute bias
    abs_bias = np.nanmean(np.abs(bias_mat), axis=0)
    Monte_carlo_abs_bias_SE = np.nanstd(np.abs(bias_mat), axis=0) / np.sqrt(n_sim - 1)

    # calculate the MSE and monte carlo SE of the MSE
    MSE = (np.nanmean(bias_mat**2, axis=0))
    Monte_carlo_SE_MSE = np.nanstd(bias_mat**2, axis=0) / np.sqrt(n_sim - 1)

    # empirical standard error
    empirical_SE = np.nanstd(bias_mat, axis=0, ddof=1)
    Monte_carlo_SE_empirical = empirical_SE / np.sqrt(2*(n_sim - 1))

    # calculate the time 
    time = np.mean(time_mat, axis=0)

    return bias, Monte_carlo_SE_bias, prct_bias, abs_bias, Monte_carlo_abs_bias_SE, Monte_carlo_SE_prct_bias, MSE, Monte_carlo_SE_MSE, empirical_SE, Monte_carlo_SE_empirical, time, nancount

def info_df(n_conf, n_observations, share_omit,   B_T, gamma, lam,
            iterations_monte_carlo, m_ML, ml_algo, linear_pi, linear_di, linear,
            random_omit , real_world_matrix = None,
            outcome_nuisance = XGBRegressor(), propensity_nuisance = RandomForestClassifier()):
    """
    dataframe that contains the information of one simulation run for all ml algorithms
    
    """
    if len(ml_algo) != m_ML:
        raise ValueError('Number of machine learning algorithms does not match the number of rows in the dataframe')
    

    # create empty info dataframe with m machine learning algorithms rows
    info_data =pd.DataFrame(columns = ['ml_algo','bias', 'Monte_carlo_SE_bias', 'prct_bias', 'Monte_carlo_SE_prct_bias',
                                        'abs_bias', 'Monte_carlo_abs_bias_SE',
                                        'MSE', 'Monte_carlo_SE_MSE', 'empirical_SE', 'Monte_carlo_SE_empirical','time', 'nancount',
                                        'n_conf', 'n_observations', 'iterations_monte_carlo', 
                                        'share_omit', 'B_T', 'gamma', 'lam', 'linear_pi', 'linear_di', 'linear',
                                        'random_omit',
                                        'outcome_nuisance', 'propensity_nuisance'],
                           index = range(m_ML))
    
    monte_carlo = monte_carlo_simulation(n_conf = n_conf, n_observations = n_observations, 
                                    iterations_monte_carlo = iterations_monte_carlo, m_ML = m_ML, 
                                    share_omit = share_omit, B_T = B_T, gamma = gamma, lam = lam, linear_pi = linear_pi, linear_di = linear_di, 
                                    linear = linear, random_omit = random_omit, real_world_matrix = real_world_matrix, 
                                    outcome_nuisance = outcome_nuisance, propensity_nuisance = propensity_nuisance)
    time_mat = monte_carlo[1]
    bias_mat = monte_carlo[0]
    
    # get measure concerning the bias and time
    bias, Monte_carlo_SE_bias, prct_bias, abs_bias, Monte_carlo_abs_bias_SE, Monte_carlo_SE_prct_bias, MSE, Monte_carlo_SE_MSE, empirical_SE, Monte_carlo_SE_empirical, time, nancount = bias_time_aggregation(bias_mat, B_T, time_mat)

    bias_wide = pd.DataFrame({'ml_algo': ml_algo})
    bias_wide = pd.concat([bias_wide, pd.DataFrame(bias_mat.T)], axis=1)
    # Convert from wide to long format using melt
    df_long = pd.melt(bias_wide, id_vars=['ml_algo'], var_name='iteration', value_name='bias')

    df_long['gamma'] = gamma
    df_long['linear_di'] = linear_di
    df_long['linear'] = linear
    df_long['lam'] = lam
    df_long['share_omit'] = share_omit
    df_long['n_observations'] = n_observations

    df_backup = df_long


    # insert the result from bias aggregation into the dataframe
    info_data['ml_algo'] = ml_algo
    info_data['bias'] = bias
    info_data['Monte_carlo_SE_bias'] = Monte_carlo_SE_bias
    info_data['prct_bias'] = prct_bias
    info_data['abs_bias'] = abs_bias
    info_data['Monte_carlo_abs_bias_SE'] = Monte_carlo_abs_bias_SE
    info_data['Monte_carlo_SE_prct_bias'] = Monte_carlo_SE_prct_bias
    info_data['MSE'] = MSE
    info_data['Monte_carlo_SE_MSE'] = Monte_carlo_SE_MSE
    info_data['empirical_SE'] = empirical_SE
    info_data['Monte_carlo_SE_empirical'] = Monte_carlo_SE_empirical
    info_data['time'] = time
    info_data['nancount'] = nancount
    info_data['n_conf'] = n_conf
    info_data['n_observations'] = n_observations
    info_data['iterations_monte_carlo'] = iterations_monte_carlo
    info_data['share_omit'] = share_omit
    info_data['B_T'] = B_T
    info_data['gamma'] = gamma
    info_data['lam'] = lam
    info_data['linear_pi'] = linear_pi
    info_data['linear_di'] = linear_di
    info_data['linear'] = linear
    info_data['random_omit'] = random_omit
    info_data['outcome_nuisance'] = str(outcome_nuisance).split('(')[0]
    info_data['propensity_nuisance'] = str(propensity_nuisance).split('(')[0]

    return info_data, df_backup


# hard code function to adjust monte carlo iterations based on needs
def monte_carlo_adjuster(n_observations, iterations_monte_carlo):
                if n_observations == 2000:
                    return 500
                elif n_observations == 8000:
                    return 300
                else:
                    print('watch out no monte carlo adjustment done')
                    return iterations_monte_carlo


# define function that combines the different simulations runs data frames
def final_df(n_conf_array, n_observations_array,
            share_omit_array, B_T_array, gamma_array, lam_array,
            iterations_monte_carlo, m_ML, ml_algo, linear_pi , linear_di, 
            linear,  random_omit, real_world_matrix = None, 
            outcome_nuisance = XGBRegressor(), propensity_nuisance = RandomForestClassifier(), 
            log = True , backup = False, breaker = False, monte_carlo_adjustment = False):
    """
    
    """
    def transfrom_to_array(*args):
        for arg in args:
            result = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    result.append(arg)
                else:
                    result.append(np.array([arg]))
            return result

    array_variables = [n_conf_array, n_observations_array, share_omit_array,  B_T_array, gamma_array,lam_array]
    n_conf_array, n_observations_array, share_omit_array,   B_T_array, gamma_array , lam_array = transfrom_to_array(*array_variables)
    array_variables = [n_conf_array, n_observations_array, share_omit_array,  B_T_array, gamma_array, lam_array]

    main_df = pd.DataFrame(columns = ['ml_algo','bias', 'Monte_carlo_SE_bias', 'prct_bias', 'Monte_carlo_SE_prct_bias',
                                        'abs_bias', 'Monte_carlo_abs_bias_SE',
                                        'MSE', 'Monte_carlo_SE_MSE', 'empirical_SE', 'Monte_carlo_SE_empirical','time', 'nancount',
                                        'n_conf', 'n_observations', 'iterations_monte_carlo', 
                                        'share_omit', 'B_T', 'gamma', 'lam', 'linear_pi', 'linear_di', 'linear',
                                        'random_omit',
                                        'outcome_nuisance', 'propensity_nuisance'])
    df_backup = pd.DataFrame(columns = ['ml_algo', 'iteration', 'bias', 'gamma', 'linear_di', 'linear', 'lam', 'share_omit', 'n_observations'])
    
    
    function_input_dictionary = {'m_ML': m_ML,'ml_algo': ml_algo,
                                'linear_pi': linear_pi, 'linear_di': linear_di, 'linear': linear, 'random_omit': random_omit,
                                'real_world_matrix': real_world_matrix,
                                'outcome_nuisance': outcome_nuisance,
                                'propensity_nuisance': propensity_nuisance}

    # find the maximum length of the arrays
    max_length = max(len(arr) for arr in array_variables)
    
    # Initialize the matrix with NaNs
    matrix = np.full((len(array_variables), max_length), np.nan, dtype='<U32')

    # Fill the matrix with array elements
    for i, arr in enumerate(array_variables):
        matrix[i, :len(arr)] = arr.astype(str)

    counter = 1

    # seem to work better but broke down in T learner
    def safe_int_conversion(x):
        try:
            return int(x)
        except:
            return x 

    #comb_list = []

    # Create all combinations of elements where each combination
    # takes one element from each row
    # combination is a tuple which can be accessed by index
    for combination in itertools.product(*matrix):

        # skip if any nan is in the combination
        if 'nan' in combination:
            continue
        print ('combination:', combination)
        print('Processing combination:', combination)
        
        n_conf = safe_int_conversion(combination[0])
        n_observations = safe_int_conversion(combination[1])
        share_omit = combination[2] # possible bug due to integer problem
        B_T = float(combination[3])  # Assuming B_T should be used as is
        gamma = float(combination[4])  # Assuming gamma should be used as is
        lam = float(combination[5])  # Assuming lam should be used as is
        #comb_list.append(combination)
        
        # adjust monte carlo iterations
        if monte_carlo_adjustment:
            iterations_monte_carlo = monte_carlo_adjuster(n_observations, iterations_monte_carlo)

        new_df, df_backup_new = info_df(n_conf=n_conf, n_observations=n_observations,
                    share_omit=share_omit, B_T=B_T, gamma=gamma, lam = lam, iterations_monte_carlo=iterations_monte_carlo,
                    **function_input_dictionary)
        
        main_df = pd.concat([main_df, new_df], axis=0)

        df_backup = pd.concat([df_backup, df_backup_new], axis=0)


        if backup:
            if counter % 20 == 0:
                main_df.to_csv('backup_simulations\\backup_df_simulation.csv', index=False)

        if breaker & (counter == 21):
            break

        print('---------')
        print('---------')
        print('---------')
        print('number of combinations:', counter)
        counter += 1
    # reset the index
    main_df.reset_index(drop=True, inplace=True)

    df_backup.reset_index(drop=True, inplace=True)

    return main_df, df_backup



# short test final df
n_conf_array = np.array([5,3])
n_observations_array = np.array([100])
x_omit = ['X999', 'X1']
share_omit_array = np.array(translate_to_np_r(x_omit))
B_T_array = 1
gamma_array = np.array([0.1])
lam_array = np.array([0.1])
iterations_monte_carlo = 2
m_ML = 5
real_world_matrix = None
ml_algo = ["Linear Regression","DML", "T-learner", "X-learner", "XBCF"]
linear_pi = False
linear_di = True
linear = True
random_omit = True
outcome_nuisance = XGBRegressor()
propensity_nuisance = XGBClassifier()
log = True
backup = True
breaker = False
monte_carlo_adjustment = True # watch out this is hard coded in the function

test_final_df, test_backup_df = final_df(n_conf_array, n_observations_array,
            share_omit_array, B_T_array, gamma_array, lam_array, 
            iterations_monte_carlo, m_ML, ml_algo, 
            linear_pi = linear_pi, linear_di = linear_di, linear = linear,
            random_omit = random_omit, real_world_matrix = real_world_matrix,  
            outcome_nuisance = outcome_nuisance, propensity_nuisance = propensity_nuisance, 
            log = log , backup = backup, breaker=breaker, monte_carlo_adjustment=monte_carlo_adjustment)

test_final_df.to_csv('backup_simulations/test_df.csv', index=False)


## ----------------------
# measure of asymtotic correlation 

# need the group sizes

# Y = f(D,X) + u  

# function that is estimated with some omitted confounders

# Y = f*(D,X) + u*  where u* is u + f*(D,X) - f(D,X)

def translate_to_np_r(strings):
    result = []
    for s in strings:
        # Translate each string to np.r_ compatible format
        parts = []
        # Split the string into individual components (variables or ranges)
        tokens = s.split()
        
        for token in tokens:
            # If it's a range (e.g., X9-X18), convert it to slice format
            if '-' in token:
                start, end = map(lambda x: int(x[1:]), token.split('-'))  # extract numbers from X9-X18
                parts.append(f"{start}:{end + 1}")  # add 1 because end index is exclusive
            else:
                # If it's a single variable (e.g., X1), just extract the number
                parts.append(f"{int(token[1:])}")
        
        # Join all parts for np.r_ use
        result.append(f"np.r_[{', '.join(parts)}]")  # e.g., np.r_[1, 2, 9:19]
    
    return result


i = 1

def corr_calculation_non_linear(n_observation, n_conf, gamma, lam):

    B_T = 1

    for step in range(2):
        # create dataframe wit the mathematical function expression and the missing confounders 
        # first create x variables as a list and then merge all the parts into a dataframe
        if step == 0:
            x_omit = ['X999', 'X1', 'X2', 'X3', 'X17', 'X18', 'X2 X18', 'X1 X2', 'X17 X18', 'X8-X9 X11-X18',  'X19 X1-X4 X14-X18', 'X19 X1-X9', 'X1-X20', 'X1-X2 X4-X18 X20']
            x_omit_table = ['', 'X1', 'X2', 'X3', 'X17', 'X18', 'X2, X18', 'X1, X2', 'X17, X18', 'X8-X9, X11-X18',  'X19, X1-X4, X14-X18', 'X19, X1-X9, X19', 'X1-X20', 'X1-X2, X4-X18, X20']
            strengthScenario = ['No Omit', 'S', 'S', 'S', 'W', 'W', 'S W', 'S S', 'W W', '10W', '5S 5W', '10S', 'All', 'All but 2S']
            x_omit_slicer = translate_to_np_r(x_omit)
        elif step == 1:
            x_omit = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20']
            strengthScenario = np.full(len(x_omit), np.nan)  
            x_omit_slicer = translate_to_np_r(x_omit)


        x_matrix = generate_X(n_observations=n_observation, n_conf=n_conf)

        y_i, d_i, v_i = DGP(x_matrix=x_matrix, B_T=B_T, gamma=gamma, linear=False, linear_pi=False, linear_di=False, lam = lam) # also need to do one with the other lambda

        # create the functions to evaluate epsilon error
        n_x = x_matrix.shape[0]
        n_y = x_matrix.shape[1]

        x_end = n_y-1
        x_mid = int(round((n_y/2),0)) - 1

        special_term_mat = np.full((n_x, n_y), 0, dtype=float)

        special_term_mat[:, 0] = x_matrix[:, 0] * x_matrix[:, x_mid] * x_matrix[:, x_end] #X1
        special_term_mat[:, 1] = ((x_matrix[:, 1]- 3) * (x_matrix[:, x_mid] + 3)**2) / np.mean((x_matrix[:, 1]- 3) * (x_matrix[:, x_mid] + 3)**2) #X2
        special_term_mat[:, 2] = np.log(np.abs(x_matrix[:,2])) * (x_matrix[:, (x_end-1)] - 3) #X3
        # the next have no special term 
        special_term_mat[:, 9] = special_term_mat[:, 0] + special_term_mat[:, 1] #X10
        special_term_mat[:, 18] = special_term_mat[:, 2]  #X19
        special_term_mat[:, 19] = special_term_mat[:, 0] #X20


        x_single_mat = np.full((n_x, n_y), np.nan)

        i = 0
        for i in range(n_y):
            n_conf = n_y
            ns = n_conf
            c = 1
            k = np.arange(1, ns+1)
            r = c - (c/ns)*k
            
            xi = x_matrix[:,i]
            ri = r[i]
            
            base_term = np.abs(xi - ri) - 1/n_y * np.abs(xi - ri)

            x_single = gamma *(base_term + special_term_mat[:, i])

            x_single_mat[:, i] = x_single

        slicer = x_omit_slicer[7]


        e = v_i[:, np.newaxis]

        arr_cov = np.full((len(x_omit)), np.nan)
        arr_corr = np.full((len(x_omit)), np.nan)
        i = 0

        for slicer in x_omit_slicer: 
            slicer = eval(slicer)

            try:
                x_term = x_single_mat[:, slicer-1]
                x_sum = np.sum(x_term, axis=1) # there is some problem with this addition 
                e_hat = x_sum + e[:, 0] 
            except:
                x_term = 0
                e_hat = np.sum(e, axis=1)# new


            #e_hat = np.sum(e + x_term, axis=1)
            #e_hat = np.sum(x_term, axis=1)

            #e_hat = np.sum(e, axis =1)

            fxd = y_i - e_hat

            arr_cov[i] = np.abs(np.cov(fxd, e_hat)[0,1])
            arr_corr[i] = np.abs(np.corrcoef(fxd, e_hat)[0,1])
            i += 1

        if step == 0:
            df_scenarios = pd.DataFrame({'x_omit': x_omit,'x_omit_slicer': x_omit_slicer, 'strengthScenario': strengthScenario, 'x_omit_table': x_omit_table,  'cov': arr_cov, 'corr': arr_corr})
        elif step == 1:
            df_each_x = pd.DataFrame({'x_omit': x_omit, 'cov': arr_cov, 'corr': arr_corr})

    return df_scenarios, df_each_x

# calculate the correlation for the non linear function 
# order of param list: n_observation, n_conf, gamma, lam
parameter_list = [
    (1000000, 20, 0.5, 1.2),
    (1000000, 20, 0.5, 0.5),
    (1000000, 20, 0.1, 0.1) 
]
df_scenario_union = pd.DataFrame()
df_each_x_union = pd.DataFrame()

corr_calculation_non_linear(*parameter_list[0])
for param in parameter_list:
    df_scenarios, df_each_x = corr_calculation_non_linear(*param)
    df_scenarios['gamma'] = param[2]
    df_scenarios['lam'] = param[3]
    df_scenarios['linear'] = False
    df_scenario_union = pd.concat([df_scenario_union, df_scenarios], axis=0)
    df_each_x['gamma'] = param[2]
    df_each_x['lam'] = param[3]
    df_scenarios['linear'] = False
    df_each_x_union = pd.concat([df_each_x_union, df_each_x], axis=0)


df_scenario_union_non_linear = df_scenario_union
df_each_x_union_non_linear = df_each_x_union

safe_df = df_scenario_union_non_linear.rename(columns = {'strengthScenario': 'Scenario', 'corr': 'Correlation', 'x_omit_slicer':'share_omit'}, inplace=False)
safe_df.to_csv('simulation/correlation/df_scenario_union_non_linear.csv', index=False)

# go for gamma == 0.5 and lambda == 0.5, 1.2

# create a table for the section with omitting confounders
df_int = corr_calculation_non_linear(1000000, 20, 0.5, 0.5)
df_scenarios = df_int[0]

df_scenarios = df_scenarios[['x_omit_table','strengthScenario','corr']]

df_scenarios = df_scenarios.rename(columns = {'strengthScenario': 'Scenario', 'corr': 'Correlation', 'x_omit_table': 'Omitted Confounders'})

# Convert DataFrame to LaTeX
latex_table = df_scenarios.to_latex(label='tab:ScenariosOmitting',
                                    caption='Different Scenarios of Confounding Omitting', 
                                    index=False)

# Add the small text right after the table and before \end{table}
custom_text = "\\\\\small{Values in Omitted Confounders correspond to confounders position $Xj$ where j = \{1,\dots,P\}. In Scenario, S and W correspond to Strong and Weak. The numbers in front of letters indicate the number of variables omitted. I.e. 5S meaning 5 strong variables. The specifications for DGP are: Non-Linear DGP, \gamma = 0.5, \lambda = 0.5}"

# Combine the LaTeX code for the table and the custom text
latex_table_with_text = latex_table.replace("\\end{table}", custom_text + "\n\\end{table}")

# Write the combined LaTeX to a file
with open(f"tables/ScenariosOmitting.tex", "w") as f:
    f.write(latex_table_with_text)


def corr_calculation_linear(n_observation, n_conf, gamma, lam):

    B_T = 1

    # create dataframe wit the mathematical function expression and the missing confounders 
    # first create x variables as a list and then merge all the parts into a dataframe
    for step in range(2):

        if step == 0:

            x_omit = ['X999', 'X1', 'X2', 'X15', 'X16', 'X1 X16', 'X1 X2', 'X15 X16', 'X10-X19', 'X1-X5 X15-X19', 'X1-X10', 'X1-X20', 'X3-X20']
            x_omit_table = ['', 'X1', 'X2', 'X15', 'X16', 'X1, X16', 'X1, X2', 'X15, X16', 'X10-X19', 'X1-X5, X15-X19', 'X1-X10', 'X1-X20', 'X3-X20']
            strengthScenario = ['No Omit', 'S', 'S', 'W', 'W', 'S W', 'S S', 'W W', '10W', '5S 5W', '10S', 'All', 'All but 2S']
            x_omit_slicer = translate_to_np_r(x_omit)

        elif step == 1:
            x_omit = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20']
            strengthScenario = np.full(len(x_omit), np.nan)  
            x_omit_slicer = translate_to_np_r(x_omit)


        x_matrix = generate_X(n_observations=n_observation, n_conf=n_conf)

        y_i, d_i, v_i = DGP(x_matrix=x_matrix, B_T=B_T, gamma=gamma, linear=True, linear_pi=False, linear_di=False, lam = lam)

        # create the functions to evaluate epsilon error
        n_x = x_matrix.shape[0]
        n_y = x_matrix.shape[1]

        x_end = n_y-1
        x_mid = int(round((n_y/2),0)) - 1
        Beta  = beta(n_y)
        x_single_mat = np.full((n_x, n_y), np.nan)

        i = 0
        for i in range(n_y):
            n_conf = n_y
            ns = n_conf
            c = 1
            k = np.arange(1, ns+1)
            r = c - (c/ns)*k
            
            xi = x_matrix[:,i]
            ri = r[i]
            
            base_term = xi * Beta[i]

            x_single = gamma *(base_term)

            x_single_mat[:, i] = x_single

        slicer = x_omit_slicer[7]


        e = v_i[:, np.newaxis]

        arr_cov = np.full((len(x_omit)), np.nan)
        arr_corr = np.full((len(x_omit)), np.nan)
        i = 0

        no_error = False

        for slicer in x_omit_slicer: 
            slicer = eval(slicer)

            if no_error: 

                try:
                    x_term = x_single_mat[:, slicer-1]
                    e_hat = np.sum(x_term, axis=1) 
                except:
                    x_term = 0
                    e_hat = np.sum(e, axis=1) 
            
            else:

                try:
                    x_term = x_single_mat[:, slicer-1]
                    x_sum = np.sum(x_term, axis=1) # there is some problem with this addition 
                    e_hat = x_sum + e[:, 0] 
                except:
                    x_term = 0
                    e_hat = np.sum(e + x_term, axis=1) 
            #e_hat = np.sum(x_term, axis=1)

            #e_hat = np.sum(e, axis =1)

            fxd = y_i - e_hat
            
            arr_cov[i] = np.abs(np.cov(fxd, e_hat)[0,1])
            arr_corr[i] = np.abs(np.corrcoef(fxd, e_hat)[0,1])
            i += 1
        if step == 0:
            df_scenarios = pd.DataFrame({'x_omit': x_omit,'x_omit_slicer': x_omit_slicer, 'strengthScenario': strengthScenario, 'x_omit_table': x_omit_table,  'cov': arr_cov, 'corr': arr_corr})
        elif step == 1:
            df_each_x = pd.DataFrame({'x_omit': x_omit, 'cov': arr_cov, 'corr': arr_corr})
    
    return df_scenarios, df_each_x

parameter_list = [
    (1000000, 20, 0.5, 0.1),
    (1000000, 20, 0.5, 0.5),
    (1000000, 20, 0.1, 0.1)
]
df_scenario_union = pd.DataFrame()
df_each_x_union = pd.DataFrame()

for param in parameter_list:
    df_scenarios, df_each_x = corr_calculation_linear(*param)
    df_scenarios['gamma'] = param[2]
    df_scenarios['lam'] = param[3]
    df_scenarios['linear'] = True
    df_scenario_union = pd.concat([df_scenario_union, df_scenarios], axis=0)
    df_each_x['gamma'] = param[2]
    df_each_x['lam'] = param[3]
    df_scenarios['linear'] = True
    df_each_x_union = pd.concat([df_each_x_union, df_each_x], axis=0)

df_scenario_union_non_linear = df_scenario_union
df_each_x_union_non_linear = df_each_x_union

df_concat = pd.concat([df_scenario_union_non_linear, df_scenario_union], axis=0)

safe_df = df_concat.rename(columns = {'strengthScenario': 'Scenario', 'corr': 'Correlation', 'x_omit_slicer':'share_omit'}, inplace=False)
safe_df.to_csv('simulations/correlation/df_scenarios.csv', index=False)


# scenario mapping to confounders for the appendix

# create a table for the section with omitting confounders
df_int = corr_calculation_linear(100000, 20, 0.5, 0.5)
df_scenarios1 = df_int[0]

df_scenarios1 = df_scenarios1[['x_omit_table','strengthScenario', 'corr']]

df_scenarios1 = df_scenarios1.rename(columns = {'strengthScenario': 'Scenario', 'x_omit_table': 'Omitted Conf.', 'corr': 'Corr'})

line = pd.DataFrame({'Scenario':'', 'Omitted Conf.': '', 'Corr': ''}, index=[3])

df_scenarios1 = pd.concat([df_scenarios1.iloc[:3], line, df_scenarios1.iloc[3:]], axis=0)

df_scenarios1.index = range(len(df_scenarios1))

df_int = corr_calculation_non_linear(100000, 20, 0.5, 0.5)
df_scenarios2 = df_int[0]

df_scenarios2 = df_scenarios2[['x_omit_table','strengthScenario', 'corr']]

df_scenarios2 = df_scenarios2.rename(columns = {'strengthScenario': 'Scenario', 'x_omit_table': 'Omitted Conf.', 'corr': 'Corr'})



# Concatenate both DataFrames side by side with a vertical separator
df_combined = pd.concat([df_scenarios1, df_scenarios2], axis=1, keys=['Linear DGP', 'Non-Linear DGP'])


# Combine the two DataFrames side-by-side
df_combined = pd.concat([df_scenarios1, df_scenarios2], axis=1)

# Define MultiIndex for the columns to include "Linear DGP" and "Non-Linear DGP"
df_combined.columns = pd.MultiIndex.from_product(
    [['Linear DGP', 'Non-Linear DGP'], df_scenarios1.columns]
)

# Export to LaTeX with custom column formatting
latex_table = df_combined.to_latex(
    multicolumn=True,
    multirow=True,
    column_format="lll|lll",  # Vertical line separator between Linear and Non-Linear
    header=True,
    index=False,
    label='tab:AppendixScenariosOmitting',
    caption='Appendix III: Different Scenarios of Confounding Omitting',
)

# Find the insertion points in the LaTeX table string
to_point = latex_table.find('toprule\n')
from_point = latex_table.find('midrule\n')

# Define the text to insert, with cmidrule for separation
text = (
    'toprule\n'
    '\\textbf{Linear DGP} &       &       & \\textbf{Non-Linear DGP} &       &       \\\\\n'
    '\\cmidrule{1-3} \\cmidrule{4-6}\n'
    'Omitted Conf. & Scenario & Corr & Omitted Conf. & Scenario & Corr \\\\\n\\'
)

# Combine the parts into the final LaTeX table string
new_latex_table = latex_table[:to_point] + text + latex_table[from_point:]

# Add the small text right after the table and before \end{table}
custom_text = "\\\\\small{Values in Omitted Conf correspond to confounders position $Xj$ where $j = \{1,\dots,P\}$. In Scenario, S and W correspond to Strong and Weak. Corr. is correlelation measure The numbers in front of letters indicate the number of variables omitted. I.e. 5S meaning 5 strong variables. The specifications for both DGPs are: $\gamma = 0.5$, $\lambda = 0.5$}"

# Combine the LaTeX code for the table and the custom text
latex_table_with_text = new_latex_table.replace("\\end{table}", custom_text + "\n\\end{table}")

# Write the combined LaTeX to a file
with open(f"tables/AppendixScenariosOmitting.tex", "w") as f:
    f.write(latex_table_with_text)

# correlation measure and others
def calculate_smd(covariates_matrix, treatment_col):
    smd_values = np.full((len(covariates_matrix[0])), np.nan)
    
    for position in range(len(covariates_matrix[0])):
        covariate_col = covariates_matrix[:,position]
        treated = covariate_col[np.where(treatment_col == 1)]
        untreated = covariate_col[np.where(treatment_col == 0)]
        
        mean_treated = np.mean(treated)
        mean_untreated = np.mean(untreated)
        
        var_treated = np.var(treated)
        var_untreated = np.var(untreated)
        
        pooled_std = np.sqrt((var_treated + var_untreated) / 2)
        
        smd = (mean_treated - mean_untreated) / pooled_std
        smd_values[position] = smd
    
    return smd_values


def linear_DGP_correlation_measure(x_matrix_real_world, iterations_monte_carlo, arr_omitting_steps, gamma, linear_di, linear_pi, n_obs, n_conf, linear = True, lam = 1.0):
    

    shape = (iterations_monte_carlo, n_conf)

    cor_matrix_XY = np.full(shape, np.nan)
    cor_matrix_XD = np.full(shape, np.nan)
    #SMD = np.full(shape, np.nan)

    # R^2 measure
    #n_omitting_steps = len(arr_omitting_steps)
    #R2_matrix = np.full((iterations_monte_carlo, n_omitting_steps), np.nan)
    for iteration in range(iterations_monte_carlo):


        x_matrix_real_world = generate_X(n_observations=n_obs, n_conf=n_conf)

        # create dataset wit linear DGP
        outcome, treat, v_i = DGP(x_matrix = x_matrix_real_world, B_T = 1, gamma = gamma, linear_pi=linear_pi, linear_di=linear_di, linear=linear, lam = lam)

        cor_XY_arr = np.corrcoef(x_matrix_real_world.T, outcome, rowvar=True)[-1, :-1] # verified that this is correct
        cor_XD_arr = np.corrcoef(x_matrix_real_world.T, treat, rowvar=True)[-1, :-1] # verified that this is correct
        #SMD_arr = calculate_smd(x_matrix_real_world, treat)

        #for omit in range(n_omitting_steps):
            #share_omit = arr_omitting_steps[omit]
            #omitted_matrix = confounder_omiting(x_matrix = x_matrix_real_world, share_omit = share_omit)
            #R2_matrix[iteration, omit] = sm.OLS(outcome, omitted_matrix).fit().rsquared

        cor_matrix_XY[iteration, :] = cor_XY_arr
        cor_matrix_XD[iteration, :] = cor_XD_arr
        #SMD[iteration, :] = SMD_arr


    # calculate the aggregates mean and standard error
    cor_XY = np.nanmean(cor_matrix_XY, axis=0)
    cor_XD = np.nanmean(cor_matrix_XD, axis=0)
    #SMD = np.nanmean(SMD, axis=0)

    # standard error 
    n = iterations_monte_carlo
    se_XY = np.nanstd(cor_matrix_XY, axis=0) / np.sqrt(n - 1)
    se_XD = np.nanstd(cor_matrix_XD, axis=0) / np.sqrt(n - 1)
    #se_SMD = np.nanstd(SMD, axis=0) / np.sqrt(n - 1)

    # stack column 
    measure_matrix_aggregate = np.column_stack((cor_XY, se_XY, cor_XD, se_XD))

    return measure_matrix_aggregate#, SMD, R2_matrix, cor_matrix_XY, cor_matrix_XD

## ----------------------
# Plot the correlation of Xi with Y and D
# Create the plot
def corr_plot(measure_df, DGP = 'Non-Linear', safe = False, number = ''):
    # Use the 'science' style
    plt.style.use(['science', 'no-latex'])

    # Modify the rcParams to set the background color to white
    plt.rcParams['axes.facecolor'] = 'white'  # Background of the axes
    plt.rcParams['figure.facecolor'] = 'white'  # Background of the figure

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the positions for the bars
    indices = np.arange(1, len(measure_df)+1)
    width = 0.35

    # Plot bars for cor_XY and cor_XD with error bars
    bars1 = ax.bar(indices - width/2, measure_df['cor_XY'], width, yerr=measure_df['se_XY'], capsize=5, label='cor_XY', error_kw={'elinewidth': 1, 'capthick': 1})
    bars2 = ax.bar(indices + width/2, measure_df['cor_XD'], width, yerr=measure_df['se_XD'], capsize=5, label='cor_XD', error_kw={'elinewidth': 1, 'capthick': 1})

    # Add labels, title, and legend
    ax.set_xlabel('Xi')
    ax.set_ylabel('Correlation')
    ax.set_title('Correlation of Xi with Y and D')
    ax.set_xticks(indices)
    ax.set_xticklabels([f'X{i}' for i in range(1,len(measure_df) +1)])
    ax.legend()

    if safe:
        # Save the figure as a PDF
        fig.savefig(f'visuals/correlations{DGP}_{number}.pdf', format='pdf', bbox_inches='tight')
    else:
        # Display the plot
        plt.show()

## ----------------------
# Plot the correlation of Xi with Y and D for multiple DGPs
def corr_plot_multi(DGP_name='Non-Linear', safe=False, number='', row=3, col=1, linear=None, gamma=None, lam=None, iterations_monte_carlo=10, len_page = 11.35):
    # Reinitialize measure_dfs
    measure_dfs = []
    for i in range(len(linear)):
        # Generate DGP result and create a DataFrame for each measure
        DGP_result = linear_DGP_correlation_measure(
            x_matrix_real_world = None, iterations_monte_carlo = iterations_monte_carlo, arr_omitting_steps = None, 
            gamma=gamma[i], linear_di=False, linear_pi=False, 
            n_obs=2000, n_conf=20, linear=linear[i], lam=lam[i]
        )
        measure_df = pd.DataFrame(DGP_result, columns=['cor_XY', 'se_XY', 'cor_XD', 'se_XD'])
        measure_dfs.append(measure_df)

    # Use the 'science' style
    plt.style.use(['science', 'no-latex'])

    # Modify the rcParams to set the background color to white
    plt.rcParams['axes.facecolor'] = 'white'  # Background of the axes
    plt.rcParams['figure.facecolor'] = 'white'  # Background of the figure
    plt.rcParams['savefig.facecolor'] = 'white' 

    # Directly set the font size for different elements
    plt.rcParams.update({
        'font.size': 14,          # Base font size for all elements
        'axes.titlesize': 16,     # Title font size
        'axes.labelsize': 14,     # Axis label font size
        'xtick.labelsize': 14,    # X-tick label size
        'ytick.labelsize': 14,    # Y-tick label size
        'legend.fontsize': 14,    # Legend font size
        'figure.titlesize': 16    # Overall figure title font size
    })

    # Set the figure size for A4 with a more vertical orientation
    fig, axs = plt.subplots(row, col, figsize=(8.27, len_page))  # A4 size in inches, portrait mode minus a bit of long 
    axs = axs.flatten()  # Flatten axs to ensure compatibility with iteration

    # Plot each measure_df on its respective subplot
    for i, (measure_df, ax) in enumerate(zip(measure_dfs, axs)):
        
        # Define the positions for the bars
        indices = np.arange(1, len(measure_df) + 1)
        width = 0.35
        
        # Plot bars for cor_XY and cor_XD with error bars
        bars1 = ax.bar(indices - width / 2, measure_df['cor_XY'], width, 
                       yerr=measure_df['se_XY'], capsize=5, label='cor_XY', 
                       error_kw={'elinewidth': 1, 'capthick': 1})
        
        bars2 = ax.bar(indices + width / 2, measure_df['cor_XD'], width, 
                       yerr=measure_df['se_XD'], capsize=5, label='cor_XD', 
                       error_kw={'elinewidth': 1, 'capthick': 1})
        
        # Add labels, title, and legend for each subplot
        ax.set_ylabel('Correlation')
        ax.set_title(f'Correlation of Xi with Y and D - gamma = {gamma[i]}, lambda = {lam[i]}')
        ax.set_xticks(indices)
        ax.set_xticklabels([f'X{i}' for i in range(1, len(measure_df) + 1)])
        ax.legend()

    # Adjust layout for readability
    plt.tight_layout()

    # Save or show the plot
    if safe:
        fig.savefig(f'visuals/correlations_{DGP_name}_{number}.pdf', format='pdf', bbox_inches='tight')
    else:
        plt.show()

# Non Linear 
linear = [False, False, False]
gamma = [0.5, 0.5, 0.1]
lam = [1.2, 0.5, 0.1]

corr_plot_multi(DGP_name='Non-Linear', safe=True, number='', row=3, col=1, linear=linear, gamma=gamma, lam=lam, iterations_monte_carlo=500)

# create plots for linear and non-linear DGP
linear = [True, True]
gamma = [0.5, 0.1]
lam = [0.5, 0.1]

corr_plot_multi(DGP_name='Linear', safe=True, number='', row=2, col=1, linear=linear, gamma=gamma, lam=lam, iterations_monte_carlo=500, len_page = 8)

## ----------------------
# same as above but for two DGPs

# create x_matrix

iterations_monte_carlo = 500

linear_DGP= linear_DGP_correlation_measure(x_matrix_real_world = None, iterations_monte_carlo = iterations_monte_carlo, arr_omitting_steps = None, gamma=0.5, 
                                                            linear_di=True, linear_pi=False, n_obs=2000, n_conf=20, linear=True, lam = 0.5)
non_linear_DGP= linear_DGP_correlation_measure(x_matrix_real_world = None, iterations_monte_carlo = iterations_monte_carlo, arr_omitting_steps = None, gamma=0.5, 
                                                            linear_di=False, linear_pi=False, n_obs=2000, n_conf=20, linear=False, lam = 1.2)

measure_list = [linear_DGP, non_linear_DGP]

# Use the 'science' style
plt.style.use(['science', 'no-latex'])

# Modify the rcParams to set the background color to white
plt.rcParams['axes.facecolor'] = 'white'  # Background of the axes
plt.rcParams['figure.facecolor'] = 'white'  # Background of the figure
plt.rcParams['savefig.facecolor'] = 'white' 

# Create 1x3 subplots
fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))  # 1 row, 3 columns

# Loop through each subplot to plot the same bars
for i in range(2):

    measure_matrix_aggregate = measure_list[i]

    measure_df = pd.DataFrame(measure_matrix_aggregate, columns=['cor_XY', 'se_XY', 'cor_XD', 'se_XD'])
    measure_df = measure_df.iloc[0:8, :]


    # Define the positions for the bars
    indices = np.arange(1, len(measure_df)+1)
    width = 0.35
    ax[i].set_facecolor('white') 

    bars1 = ax[i].bar(indices - width/2, measure_df['cor_XY'], width, yerr=measure_df['se_XY'], capsize=5, label='cor_XY', error_kw={'elinewidth': 1, 'capthick': 1})
    bars2 = ax[i].bar(indices + width/2, measure_df['cor_XD'], width, yerr=measure_df['se_XD'], capsize=5, label='cor_XD', error_kw={'elinewidth': 1, 'capthick': 1})

    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    ax[i].set_xticks(indices)
    ax[i].set_xticklabels([f'X{i}' for i in range(1, len(measure_df) + 1)])

    # Use tick_params to set fontsize
    ax[i].tick_params(axis='x', labelsize=18)
    ax[i].tick_params(axis='y', labelsize=22)

# Add the legend only in the first subplot (e.g., ax[0]) inside the plot area
ax[0].legend(loc='upper right', frameon=True, fontsize=20)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure as a PDF
fig.savefig(f'visuals/correlations.pdf', format='pdf', bbox_inches='tight')

# Show the plots
#plt.show()

plt.close()

###############################################################
# plot the propensity score, common support 
n_observations = 10000
n_conf = 20
x_matrix = generate_X(n_observations=n_observations, n_conf=n_conf)
lam_list = [0.5, 1.2]

data_list = []
for i in range(2):
    # Generate treatment variable with more overlap
    D, probabilities = D_i_generator(
        x_matrix,
        linear_di=False,
        linear=False,
        lam=lam_list[i]
    )
    
    # Create a DataFrame with propensity scores and treatment indicators
    data = pd.DataFrame({
        'propensity_score': probabilities,
        'treatment': D
    })

    data_list.append(data)

# Use the 'science' style
plt.style.use(['science', 'no-latex'])

# Modify the rcParams to set the background color to white
plt.rcParams['axes.facecolor'] = 'white'  # Background of the axes
plt.rcParams['figure.facecolor'] = 'white'  # Background of the figure
plt.rcParams['savefig.facecolor'] = 'white' 

# Directly set the font size for different elements
plt.rcParams.update({
    'font.size': 18,          # Base font size for all elements
    'axes.titlesize': 22,     # Title font size
    'axes.labelsize': 18,     # Axis label font size
    'xtick.labelsize': 18,    # X-tick label size
    'ytick.labelsize': 18,    # Y-tick label size
    'legend.fontsize': 18,    # Legend font size
    'figure.titlesize': 24    # Overall figure title font size
})

# Create 1x2 subplots
fig, ax = plt.subplots(1, 2, figsize=(18, 5.5))  # 1 row, 2 columns

# Loop through each subplot to plot the propensity score distributions
for i in range(2):
    sns.histplot(
        data=data_list[i], 
        x='propensity_score', 
        hue='treatment', 
        common_norm=False,           # Do not normalize within groups
        palette=['blue', 'orange'],
        alpha=0.5,
        element='bars',              # Draw bars
        stat='density',              # Normalize to show density rather than counts
        bins=20,                     # Adjust bin size as desired
        multiple="layer",            # Overlay bars for different groups
        ax=ax[i]                     # Use the appropriate subplot axis
    )
    ax[i].set_xlim(0, 1)             # Ensure x-axis is limited to [0, 1]
    ax[i].set_title(f'Propensity Score Distribution for $\lambda={lam_list[i]}$')
    ax[i].set_xlabel('Propensity Score')
    ax[i].set_ylabel('Density')
    ax[i].legend(title='Treatment', labels=['Treated (D=1)', 'Control (D=0)'])

# Adjust layout to fit the subplots
plt.tight_layout()

# Save the figure as a PDF
fig.savefig('propensityVisuals/lambda_comparison_main.pdf', format='pdf', bbox_inches='tight')

# Show the plots
plt.show()
plt.close()


# for the appendix

n_observations = 10000
n_conf = 20
x_matrix = generate_X(n_observations=n_observations, n_conf=n_conf)
lam_list = [0.1, 0.5, 1.2, 0.1, 0.5]
DGP_list = [False, False, False, True, True]
DGP_name_list = ['Non-Linear', 'Non-Linear', 'Non-Linear', 'Linear', 'Linear']

data_list = []
for i in range(5):
    # Generate treatment variable with more overlap
    D, probabilities = D_i_generator(
        x_matrix,
        linear_di=False,
        linear=DGP_list[i],
        lam=lam_list[i]
    )
    
    # Create a DataFrame with propensity scores and treatment indicators
    data = pd.DataFrame({
        'propensity_score': probabilities,
        'treatment': D
    })

    data_list.append(data)

# Use the 'science' style
plt.style.use(['science', 'no-latex'])

# Modify the rcParams to set the background color to white
plt.rcParams['axes.facecolor'] = 'white'  # Background of the axes
plt.rcParams['figure.facecolor'] = 'white'  # Background of the figure
plt.rcParams['savefig.facecolor'] = 'white' 

# Directly set the font size for different elements
plt.rcParams.update({
    'font.size': 18,          # Base font size for all elements
    'axes.titlesize': 22,     # Title font size
    'axes.labelsize': 18,     # Axis label font size
    'xtick.labelsize': 18,    # X-tick label size
    'ytick.labelsize': 18,    # Y-tick label size
    'legend.fontsize': 18,    # Legend font size
    'figure.titlesize': 24    # Overall figure title font size
})


# Create the figure and the GridSpec layout
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)  # 2 rows, 2 columns grid

# Create subplots using GridSpec
ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
ax2 = fig.add_subplot(gs[0, 1])  # First row, second column
ax3 = fig.add_subplot(gs[1, 0])  # Second row

# Plot on each subplot (using placeholder data)
for i, ax in enumerate([ax1, ax2, ax3]):
    sns.histplot(
        data=data_list[i],         # Placeholder data
        x='propensity_score',
        hue='treatment',
        common_norm=False,
        palette=['blue', 'orange'],
        alpha=0.5,
        element='bars',            # Use traditional bars
        stat='density',            # Show density rather than counts
        bins=20,                   # Adjust the number of bins as needed
        multiple="layer",          # Overlay bars for different groups
        ax=ax                      # Plot in the correct subplot
    )
    #ax.set_xlim(0, 1)              # Ensure x-axis is limited to [0, 1]
    ax.set_title(f'Propensity Score Distribution {DGP_name_list[i]} for $\lambda={lam_list[i]}$')
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.legend(title='Treatment', labels=['Treated (D=1)', 'Control (D=0)'])

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Optional: save the figure
fig.savefig('propensityVisuals/lambdaComparisonAppendixNonLinear.pdf', format='pdf', bbox_inches='tight')

# Show the plots
plt.show()
plt.close()


n_observations = 10000
n_conf = 20
x_matrix = generate_X(n_observations=n_observations, n_conf=n_conf)
lam_list = [ 0.1, 0.5]
DGP_list = [True, True]
DGP_name_list = ['Linear', 'Linear']

data_list = []
for i in range(5):
    # Generate treatment variable with more overlap
    D, probabilities = D_i_generator(
        x_matrix,
        linear_di=False,
        linear=DGP_list[i],
        lam=lam_list[i]
    )
    
    # Create a DataFrame with propensity scores and treatment indicators
    data = pd.DataFrame({
        'propensity_score': probabilities,
        'treatment': D
    })

    data_list.append(data)

# Use the 'science' style
plt.style.use(['science', 'no-latex'])

# Modify the rcParams to set the background color to white
plt.rcParams['axes.facecolor'] = 'white'  # Background of the axes
plt.rcParams['figure.facecolor'] = 'white'  # Background of the figure
plt.rcParams['savefig.facecolor'] = 'white' 

# Directly set the font size for different elements
plt.rcParams.update({
    'font.size': 18,          # Base font size for all elements
    'axes.titlesize': 22,     # Title font size
    'axes.labelsize': 18,     # Axis label font size
    'xtick.labelsize': 18,    # X-tick label size
    'ytick.labelsize': 18,    # Y-tick label size
    'legend.fontsize': 18,    # Legend font size
    'figure.titlesize': 24    # Overall figure title font size
})

# Create the figure and the GridSpec layout
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)  # 2 rows, 2 columns grid

# Create subplots using GridSpec
ax1 = fig.add_subplot(gs[0, 0])  # First row, first column
ax2 = fig.add_subplot(gs[0, 1])  # First row, second column

# Plot on each subplot (using placeholder data)
for i, ax in enumerate([ax1, ax2]):
    sns.histplot(
        data=data_list[i],             # Placeholder data
        x='propensity_score',
        hue='treatment',
        common_norm=False,
        palette=['blue', 'orange'],
        alpha=0.5,
        element='bars',                # Use traditional bars
        stat='density',                # Show density rather than counts
        bins=20,                       # Adjust the number of bins as needed
        multiple="layer",              # Overlay bars for different groups
        ax=ax                          # Plot in the correct subplot
    )
    ax.set_xlim(0, 1)                  # Ensure x-axis is limited to [0, 1]
    ax.set_title(f'Propensity Score Distribution {DGP_name_list[i]} for $\lambda={lam_list[i]}$')
    ax.set_xlabel('Propensity Score')
    ax.set_ylabel('Density')
    ax.legend(title='Treatment', labels=['Treated (D=1)', 'Control (D=0)'])

# Adjust the layout to prevent overlapping
plt.tight_layout()

# Save the figure as a PDF
fig.savefig('propensityVisuals/lambdaComparisonAppendixLinear.pdf', format='pdf', bbox_inches='tight')

# Show the plots
plt.show()
plt.close()