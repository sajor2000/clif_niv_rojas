import pandas as pd
import numpy as np
import os
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.base import BaseEstimator
if not hasattr(BaseEstimator, '_validate_data'):
    from sklearn.utils.validation import validate_data
    BaseEstimator._validate_data = lambda self, *a, **kw: validate_data(self, *a, **kw)
from firthlogist import FirthLogisticRegression


# Resolve paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Load site config
config_path = os.path.join(ROOT_DIR, 'config.json')
if not os.path.exists(config_path):
    config_path = os.path.join(ROOT_DIR, 'clif_config.json')
with open(config_path) as f:
    config = json.load(f)
SITE = config.get('site', 'unknown')

nippv_data = pd.read_csv(os.path.join(ROOT_DIR, 'output', 'NIPPV_analytic_dataset.csv'))

 

# Define the list of predictors for univariate logistic regression

predictors = [

    'age_scale', 'female', 'pco2_scale', 'ph_scale', 'map_scale',
    'rr_scale', 'hr_scale', 'fio2_high', 'peep_scale', 'tidal_volume_scale'
]
 

# Initialize list to store univariate results

univariate_results = []

 

# Univariate logistic regression for each predictor

for predictor in predictors:

    try:

        # Define the formula for univariate logistic regression

        formula = f'failure ~ {predictor}'

       

        # Fit the logistic regression model

        model = smf.logit(formula=formula, data=nippv_data).fit(disp=0)

       

        # Get OR, p-value, and CI for the predictor (named indexing)

        odds_ratio = np.exp(model.params[predictor])

        p_value = model.pvalues[predictor]

        conf_int = model.conf_int()

        lower_ci = np.exp(conf_int.loc[predictor, 0])

        upper_ci = np.exp(conf_int.loc[predictor, 1])

       

        # Store the results (including log-OR and SE for meta-analysis)

        univariate_results.append({

            'Variable': predictor,

            'log_OR': model.params[predictor],

            'SE_log_OR': model.bse[predictor],

            'Odds Ratio': odds_ratio,

            'P-Value': p_value,

            '95% CI Lower': lower_ci,

            '95% CI Upper': upper_ci

        })

       

    except Exception as e:

        print(f"Error processing variable {predictor}: {e}")

 

# Convert univariate results to DataFrame

univariate_results_df = pd.DataFrame(univariate_results)

 

# Display univariate results

print("Univariate Logistic Regression Results:")

print(univariate_results_df)

 

# Define the formula for the final multivariable model with specific interaction terms

final_formula = ('failure ~ age_scale + female + pco2_scale + ph_scale + map_scale'

                 '+ rr_scale + hr_scale + fio2_high + peep_scale + tidal_volume_scale')

# Fit the final multivariable logistic regression model

final_model = smf.logit(formula=final_formula, data=nippv_data).fit()

 

# Extract ORs, p-values, CIs, and SEs for multivariable model

multivariable_results = pd.DataFrame({

    'Variable': final_model.params.index,

    'log_OR': final_model.params.values,

    'SE_log_OR': final_model.bse.values,

    'Odds Ratio': np.exp(final_model.params.values),

    'P-Value': final_model.pvalues.values,

    '95% CI Lower': np.exp(final_model.conf_int()[0].values),

    '95% CI Upper': np.exp(final_model.conf_int()[1].values)

})



# Display multivariable results

print("\nMultivariable Logistic Regression Results:")

print(multivariable_results)

# =====================================================
# MODEL DIAGNOSTICS
# =====================================================

# Variance Inflation Factors (multicollinearity check)
X_vif = nippv_data[predictors].dropna()
vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
print("\nVariance Inflation Factors:")
print(vif_data.to_string(index=False))

# AUC / c-statistic
predicted_probs = final_model.predict(nippv_data)
auc = roc_auc_score(nippv_data['failure'], predicted_probs)
print(f"\nAUC (c-statistic): {auc:.4f}")

# AUC Standard Error — Hanley-McNeil approximation
n1 = int(nippv_data['failure'].sum())    # events (positive cases)
n0 = len(nippv_data) - n1                # non-events
q1 = auc / (2 - auc)
q2 = (2 * auc**2) / (1 + auc)
auc_var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - auc**2)) / (n1 * n0)
auc_se = np.sqrt(auc_var)
print(f"AUC SE (Hanley-McNeil): {auc_se:.4f}")

# Calibration metrics (TRIPOD+AI required)
linear_pred = final_model.model.exog @ final_model.params  # log-odds
cal_model = sm.GLM(
    nippv_data['failure'],
    sm.add_constant(linear_pred),
    family=sm.families.Binomial()
).fit()
cal_intercept = cal_model.params.iloc[0]   # calibration-in-the-large (ideal=0)
cal_slope = cal_model.params.iloc[1]       # calibration slope (ideal=1)
brier = brier_score_loss(nippv_data['failure'], predicted_probs)
print(f"Apparent calibration intercept: {cal_intercept:.4f}")
print(f"Apparent calibration slope: {cal_slope:.4f}")
print(f"Apparent Brier score: {brier:.4f}")

# Bootstrap optimism-corrected internal validation (Harrell 1996, Steyerberg 2019)
print("\nBootstrap internal validation (200 iterations)...")
n_boot = 200
rng = np.random.default_rng(42)
n_obs = len(nippv_data)
optimism_auc, optimism_slope, optimism_brier = [], [], []

for b in range(n_boot):
    idx = rng.integers(0, n_obs, size=n_obs)
    boot_data = nippv_data.iloc[idx].reset_index(drop=True)
    try:
        boot_model = smf.logit(formula=final_formula, data=boot_data).fit(disp=0)
        # Training performance (on bootstrap sample)
        p_train = boot_model.predict(boot_data)
        auc_train = roc_auc_score(boot_data['failure'], p_train)
        brier_train = brier_score_loss(boot_data['failure'], p_train)
        lp_train = np.log(np.clip(p_train, 1e-10, 1 - 1e-10) / (1 - np.clip(p_train, 1e-10, 1 - 1e-10)))
        cal_train = sm.GLM(boot_data['failure'], sm.add_constant(lp_train),
                           family=sm.families.Binomial()).fit()
        slope_train = cal_train.params.iloc[1]
        # Test performance (on original data)
        p_test = boot_model.predict(nippv_data)
        auc_test = roc_auc_score(nippv_data['failure'], p_test)
        brier_test = brier_score_loss(nippv_data['failure'], p_test)
        lp_test = np.log(np.clip(p_test, 1e-10, 1 - 1e-10) / (1 - np.clip(p_test, 1e-10, 1 - 1e-10)))
        cal_test = sm.GLM(nippv_data['failure'], sm.add_constant(lp_test),
                          family=sm.families.Binomial()).fit()
        slope_test = cal_test.params.iloc[1]
        optimism_auc.append(auc_train - auc_test)
        optimism_slope.append(slope_train - slope_test)
        optimism_brier.append(brier_train - brier_test)
    except Exception:
        continue

if optimism_auc:
    corrected_auc = auc - np.mean(optimism_auc)
    corrected_slope = cal_slope - np.mean(optimism_slope)
    corrected_brier = brier - np.mean(optimism_brier)
    print(f"Bootstrap iterations completed: {len(optimism_auc)}/{n_boot}")
    print(f"Optimism-corrected AUC: {corrected_auc:.4f} (optimism: {np.mean(optimism_auc):.4f})")
    print(f"Optimism-corrected cal slope: {corrected_slope:.4f} (optimism: {np.mean(optimism_slope):.4f})")
    print(f"Optimism-corrected Brier: {corrected_brier:.4f} (optimism: {np.mean(optimism_brier):.4f})")
else:
    corrected_auc = auc
    corrected_slope = cal_slope
    corrected_brier = brier
    print("WARNING: All bootstrap iterations failed — using apparent values")

# Events per variable (EPV) — TRIPOD+AI required
n_predictors = len(predictors)
epv = n1 / n_predictors
print(f"EPV: {epv:.1f} ({n1} events / {n_predictors} predictors)")
if epv < 10:
    print(f"WARNING: EPV = {epv:.1f} is below recommended threshold of 10 (TRIPOD+AI)")

diagnostics = pd.DataFrame({
    'site': [SITE],
    'model': ['multivariable_no_interaction'],
    'N': [len(nippv_data)],
    'N_events': [n1],
    'AUC': [round(auc, 4)],
    'AUC_SE': [round(auc_se, 6)],
    'AUC_corrected': [round(corrected_auc, 4)],
    'cal_intercept': [round(cal_intercept, 4)],
    'cal_slope': [round(cal_slope, 4)],
    'cal_slope_corrected': [round(corrected_slope, 4)],
    'brier_score': [round(brier, 4)],
    'brier_corrected': [round(corrected_brier, 4)],
    'EPV': [round(epv, 1)],
    'n_predictors': [n_predictors],
    'log_likelihood': [final_model.llf],
    'AIC': [final_model.aic],
    'BIC': [final_model.bic],
    'pseudo_r2': [final_model.prsquared],
    'converged': [final_model.mle_retvals['converged']]
})

# =====================================================
# ADD SITE METADATA TO ALL RESULTS
# =====================================================

N = len(nippv_data)
N_events = int(nippv_data['failure'].sum())

univariate_results_df['site'] = SITE
univariate_results_df['N'] = N
univariate_results_df['N_events'] = N_events
univariate_results_df['model_type'] = 'univariate'

multivariable_results['site'] = SITE
multivariable_results['N'] = N
multivariable_results['N_events'] = N_events
multivariable_results['model_type'] = 'multivariable_no_interaction'

# =====================================================
# EXPORT RESULTS TO CSV
# =====================================================

SHARE_DIR = os.path.join(ROOT_DIR, 'output_to_share')
os.makedirs(SHARE_DIR, exist_ok=True)

univariate_results_df.to_csv(os.path.join(SHARE_DIR, "univariate_logistic_results.csv"), index=False)

multivariable_results.to_csv(os.path.join(SHARE_DIR, "multivariable_logistic_results_NoInteraction.csv"), index=False)

diagnostics.to_csv(os.path.join(SHARE_DIR, "model_diagnostics_NoInteraction.csv"), index=False)

# Export VIF
vif_data['site'] = SITE
vif_data.to_csv(os.path.join(SHARE_DIR, "vif_NoInteraction.csv"), index=False)

# =====================================================
# VARIANCE-COVARIANCE MATRIX (for multivariate pooling)
# =====================================================
vcov = final_model.cov_params()
vcov.to_csv(os.path.join(SHARE_DIR, "vcov_matrix_NoInteraction.csv"))
print(f"VCV matrix exported: {vcov.shape[0]}x{vcov.shape[1]}")

# =====================================================
# FIRTH'S PENALIZED LOGISTIC REGRESSION (sensitivity)
# =====================================================
print("\n--- Firth's Penalized Logistic Regression (Sensitivity Analysis) ---")
X_firth = nippv_data[predictors].values
y_firth = nippv_data['failure'].values

try:
    firth = FirthLogisticRegression(max_iter=250)
    firth.fit(X_firth, y_firth)

    firth_coefs = np.concatenate([[firth.intercept_], firth.coef_]).flatten()
    firth_vars = ['Intercept'] + predictors

    firth_results = pd.DataFrame({
        'Variable': firth_vars,
        'log_OR': firth_coefs,
        'SE_log_OR': firth.bse_,
        'Odds Ratio': np.exp(firth_coefs),
        'P-Value': firth.pvals_,
        '95% CI Lower': np.exp(firth.ci_[:, 0]),
        '95% CI Upper': np.exp(firth.ci_[:, 1]),
        'site': SITE,
        'N': N,
        'N_events': N_events,
        'model_type': 'firth_no_interaction'
    })

    firth_results.to_csv(os.path.join(SHARE_DIR, "firth_multivariable_results_NoInteraction.csv"), index=False)
    print("Firth results exported.")
    print(firth_results[['Variable', 'Odds Ratio', 'P-Value']].to_string(index=False))
except Exception as e:
    print(f"WARNING: Firth regression failed: {e}")
    print("MLE results remain the primary analysis.")

print(f"\nResults exported for site: {SITE} (N={N}, events={N_events})")
