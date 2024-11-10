# DSC-0001: Use Pydantic
# Date: 17-10-2024
# Decision: Use Pydantic
# Status: Accepted
# Motivation: No strict typing in Python
# Reason: Pydantic can enforce strict typing and encapsulation 
# Limitations: Certain types like nparrays do not work as PrivateAttr without a config dict
# Alternatives: marshmallow
