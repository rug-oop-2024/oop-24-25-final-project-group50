# DSC-0013: Replace Lasso
# Date: 6-11-2024
# Decision: Replace sklearn's Lasso with sklearn's LassoCV
# Status: Accepted
# Motivation: LassoCV automatically adjusts the alpha parameter
# Reason: Lasso is not suitable for all datasets, LassoCV preforms better
# Limitations: LassoCV is a bit slower than Lasso
# Alternatives: Other regression models
