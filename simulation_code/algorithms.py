##############################################################################
# ml_algorithms


import numpy as np
import pandas as pd

# statistical models packages
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from doubleml import DoubleMLIRM, DoubleMLPLR, DoubleMLData
from xgboost import XGBRegressor, XGBClassifier
from xbcausalforest import XBCF


# meta learner packages
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseRRegressor, BaseXRegressor, TMLELearner
model = LinearRegression()
model.fit



def nuisance_pot_outcome(outcome, treat, exog, estimator_algorithm = 'RF_classifier', classification = False):
    # get only exogenous variables where treatment == 0
    exog0 = exog[np.where(treat == 0)]
    outcome0 = outcome[np.where(treat == 0)]

    # get only exogenous where treatment == 1
    exog1 = exog[np.where(treat == 1)]
    outcome1 = outcome[np.where(treat == 1)]

    if estimator_algorithm == 'OLS':
        # fit the model with linear regression

        # fit the model
        model = LinearRegression(outcome, exog).fit()
        # get the prediction
        pred = model.predict(exog)

    if classification:
        if estimator_algorithm == 'RF_classifier':
            # define model for treatment == 0
            model0 = RandomForestClassifier(n_estimators=100, max_depth=10)
            model0.fit(exog0,outcome0)

            # define model for treatment == 1
            model1 = RandomForestClassifier(n_estimators=100, max_depth=10)
            model1.fit(exog1,outcome1)

    else:
        if estimator_algorithm == 'RF_regressor':
            # define model for treatment == 0
            model0 = RandomForestRegressor(n_estimators=100, max_depth=10)
            model0.fit(exog0,outcome0)

            # define model for treatment == 1
            model1 = RandomForestRegressor(n_estimators=100, max_depth=10)
            model1.fit(exog1,outcome1)

    # predict the potential outcome for treatment == 0
    mu0 = model0.predict(exog)

    # predict the potential outcome for treatment == 1
    mu1 = model1.predict(exog)

    return mu0, mu1


def nuisance_propensity_score(treat, exog, estimator_algorithm = XGBClassifier()):

    model = estimator_algorithm

    # fit the model
    model.fit(exog, treat)

    # get the probability of being treated
    prob = model.predict_proba(exog)[:, 1]

    # ensure that probability is not zero nore one
    prob = np.clip(prob, 0.05, 0.95)

    return prob

def lin_reg(treat, outcome, exog, propensity_ML = None, outcome_ML = None, binary_treatment = None, n_folds = None):
    treat_trans = treat.reshape(-1,1)
    stacked = np.hstack([treat_trans, exog])
    stacked = sm.add_constant(stacked)
    model = sm.OLS(outcome, stacked).fit()
    ate = model._results.params[1]
    return ate

# algorithm 1
# own IPW and AIPW implementation
def aipw_ipw_formula(treat, outcome, prob, normalize=True, doubly_robust=False, mu0 = None, mu1 = None):
    """
    IPW and AIPW formula.

    Parameters
    ----------
    treat : TYPE: pd.Series or array
        DESCRIPTION: vector of treatments
    outcome : TYPE: pd.Series or array
        DESCRIPTION: vector of outcomes
    exog : TYPE: pd.DataFrame or array
        DESCRIPTION: matrix of covariates
    prob : TYPE: pd.Series / array
        DESCRIPTION: vector of probabilities
    normalize : TYPE: bool, optional
        DESCRIPTION: whether to normalize the weights. The default is True.
    doubly_robust : TYPE: bool, optional
        DESCRIPTION: whether to use doubly robust version. The default is False.
    mu0 : TYPE: pd.Series or array, optional
        DESCRIPTION: potential outcome for treatment == 0. The default is None.
    mu1 : TYPE: pd.Series or array, optional
        DESCRIPTION: potential outcome for treatment == 1. The default is None.
    Returns
    -------
    result: ATE estimate according to IPW
    """
    if doubly_robust == True:

        ate = np.mean((mu1 - mu0) + (treat * (outcome - mu1) / prob) - ((1 - treat) * (outcome - mu0) / (1 - prob)))

    # compute ate based on ipw formula, take the mean using numpy
    # This is the standard version - known to have bad small sample properties
    if normalize == False and doubly_robust == False:
        ate = np.mean((treat * outcome) /
                      prob - ((1 - treat) * outcome) / (1 - prob))
    if normalize == True and doubly_robust == False:
        # We can achieve better small sample properties if we normalize
        # the weights for treated and controls
        w_1 = treat / prob
        w_0 = (1 - treat) / (1 - prob)
        w_1 = w_1 / np.sum(w_1)
        w_0 = w_0 / np.sum(w_0)
        ate = np.sum(w_1 * outcome) - np.sum(w_0 * outcome)
    # return the result
    return ate

# algorithm 2 AIPW with cross-fitting according to Chernozhukov et al. (2018)
# recommendation of folds accordingly to the paper is 4-5
def aipw_cv(treat, outcome, exog, propensity_ML = XGBClassifier(), outcome_ML = XGBRegressor(), binary_treatment = True, n_folds = 5):
    dml_data = DoubleMLData.from_arrays(x = exog, y = outcome, d = treat)
    
    # nuisance function for propensity score estimation 
    if propensity_ML == 'RF_classifier':
        # define model
        model_prop = RandomForestClassifier(n_estimators=100, max_depth=7)
    else:
        model_prop = propensity_ML
    # nuisance function for outcome estimation
    if outcome_ML == 'RF_regressor':
        # define model for outcome estimation
        model_outcome = RandomForestRegressor(n_estimators=100, max_depth=7)
    else:
        model_outcome = outcome_ML
    
    if binary_treatment:
        dml_model = DoubleMLIRM(dml_data, ml_m = model_prop, ml_g = model_outcome, n_folds = n_folds)
    else:
        dml_model = DoubleMLPLR(dml_data, ml_m = model_prop, ml_l= model_outcome,  ml_g = None, n_folds = n_folds)
    
    dml_model.fit()

    # get the ATE
    ate = dml_model.coef

    # get standard errors
    se = dml_model.se

    # get the confidence intervals 95%
    confidence_interval = dml_model.confint()

    # get lower bound
    lb = confidence_interval.iloc[0,0]
    ub = confidence_interval.iloc[0,1]

    # check if true ate is within the confidence interval
    if ate > lb and ate < ub:
        coveragerate = 1
    else:   
        coveragerate = 0

    return ate


# regression imputation. estimating the outcome function
# algorithm 3
# meta learner S learner form causalml
def meta_s_learner(treat, outcome, exog, propensity_ML = None, outcome_ML = XGBRegressor(), binary_treatment = None, n_folds = None):
    learner_s = BaseSRegressor(learner=outcome_ML)
    ate = learner_s.estimate_ate(X=exog, treatment=treat, y=outcome)[0]

    return ate

# algorithm 4
# meta learner T learner from causalml
def meta_t_learner(treat, outcome, exog, propensity_ML = None, outcome_ML = XGBRegressor(), binary_treatment = None, n_folds = None):
    learner_t = BaseTRegressor(learner=outcome_ML)
    ate = learner_t.estimate_ate(X=exog, treatment=treat, y=outcome)[0]

    return ate
# algorithm 5
# meta learner R learner from causalml
# propensity score is by default estimated with ElasticNetPropensityModel()
def meta_r_learner(treat, outcome, exog, propensity_ML = XGBClassifier(), outcome_ML = XGBRegressor(), binary_treatment = None, n_folds = None):
    learner_r = BaseRRegressor(learner=outcome_ML, propensity_learner=propensity_ML)
    ate = learner_r.estimate_ate(X=exog, treatment=treat, y=outcome)[0]

    return ate

# algorithm 6
# meta learner X learner from causalml
def meta_x_learner(treat, outcome, exog, propensity_ML = None, outcome_ML = XGBRegressor(), binary_treatment = None, n_folds = None):
    learner_x = BaseXRegressor(learner=outcome_ML)
    ate = learner_x.estimate_ate(X=exog, treatment=treat, y=outcome)[0]

    return ate


# algorithm 7
# X bayesian causal forest based on Hahn et al. (2020)
# do this later
def XBCF_ate(treat, outcome, exog, propensity_ML = None, outcome_ML = None, binary_treatment = None, n_folds = None):

    model = XBCF(p_categorical_pr=0,p_categorical_trt=0)
    model.fit(x_t=exog, x=exog, y=outcome, z=treat)
    tau = model.predict(X=exog)
    
    return np.mean(tau)

# algorithm 8 TMLE
def TMLE_ate(treat, outcome, exog, propensity_ML = XGBClassifier(), outcome_ML = XGBRegressor(), binary_treatment = None, n_folds = None):
    learner_tmle = TMLELearner(learner=outcome_ML) 
    ate = learner_tmle.estimate_ate(X=exog, treatment=treat, 
                            y=outcome, 
                            p = nuisance_propensity_score(treat = treat, exog = exog, 
                                                         estimator_algorithm=propensity_ML))[0]
    return ate

#TMLE_ate(treat, outcome, exog, propensity_score_algorithm = XGBClassifier())