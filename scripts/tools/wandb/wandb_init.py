try:
    import wandb
except ImportError:
    print("Warning: wandb is not installed. Please run `pip install wandb` to enable wandb support.")
    # ask user to install wandb
    install_wandb = input("Do you want to install wandb? (y/n): ")
    if install_wandb.lower() == 'y':
        import pip
        import sys
        # confirm python environment
        print(f"Current python environment: {sys.executable}")
        # ask user to confirm the python environment
        confirm_env = input("Do you want to install wandb in this python environment? (y/n): ")
        if confirm_env.lower() == 'y':
            pip.main(['install', 'wandb'])
            print("wandb is installed successfully, try to run again.")
        else:
            print("wandb is not installed.")
            
    else:
        print("wandb is not installed.")

# ask user to login wandb
import wandb
wandb_login = input("Do you want to login wandb? (y/n): ")
if wandb_login.lower() == 'y':
    try:
        wandb.login()
        print("wandb is logged in successfully.")
            
    except Exception as e:
        print(f"wandb login failed: {e}")
    
    
    
