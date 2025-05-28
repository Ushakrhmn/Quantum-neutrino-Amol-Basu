"""
A simple script to update qiskit token from command line input.
"""

import os
from qiskit_ibm_runtime import QiskitRuntimeService

def update_token(token: str, instance:str, name="default", set_as_default=True) -> None:

    QiskitRuntimeService.save_account(
        token=token,
        channel="ibm_cloud", # `channel` distinguishes between different account types.
        instance=instance, # Copy the instance CRN from the Instance section on the dashboard.
        name=name, # Optionally name this set of credentials.
        overwrite=True, # Only needed if you already have Cloud credentials.
        set_as_default=set_as_default  # Set this account as the default for future use.    
    )

if __name__ == "__main__":
    token = input("Enter your IBM Quantum API token: ")
    instance = input("Enter your IBM Quantum instance (Copy CRN from cloud platform): ")
    name = input("Enter a name for this account (default is 'default'): ") or "default"
    
    update_token(token, instance, name)
    print(f"Token updated successfully for instance '{instance}' with name '{name}'.")