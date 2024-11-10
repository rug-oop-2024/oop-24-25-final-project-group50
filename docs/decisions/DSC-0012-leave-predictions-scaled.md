# DSC-0012: Leave predictions scaled
# Date: 5-11-2024
# Decision: We do not scale back the predictions before presenting them
# Status: Accepted
# Motivation: Requires saving the scaler for pipeline saving
# Reason: Scaling back the target feature can only be done with the fitted scaler
# Limitations: Not easy to interpret predictions for user
# Alternatives: Do scale back the predictions
