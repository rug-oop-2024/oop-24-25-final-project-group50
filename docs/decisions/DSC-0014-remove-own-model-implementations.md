# DSC-0014: Remove own model implementations
# Date: 7-11-2024
# Decision: Replace our own implementations of MultipleLinearRegression and KNearestNeighbors with sklearn's models
# Status: Accepted
# Motivation: Implementations did not work well with dataset processing
# Reason: The way the data was processed could not be handled by our own implementations
# Limitations: Less transparency of code
# Alternatives: Rework our own implementations
