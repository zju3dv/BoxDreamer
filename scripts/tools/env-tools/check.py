# check workspace dependencies
import os
import sys
import argparse
import pip

try:
    import tqdm
    import loguru
except ImportError:
    print("tqdm not found. Installing...")
    pip.main(['install', 'tqdm'])
    print("loguru not found. Installing...")
    pip.main(['install', 'loguru'])

def check_workspace_dependencies(path, auto_install=False, quiet=False) -> None:
    '''
        Check workspace dependencies.
        dependencies file will be saved at python runtime path.
        
        Parameters:
            path (str): workspace path
            auto_install (bool): auto install dependencies if not found
            
        Returns:
            None
    '''
    pre_defined_dependencies = ['tensorboardX']
    # get all python files in workspace, check whether they have import errors
    python_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
                
    # get all dependencies
    dependencies = set()
    dependencies.update(pre_defined_dependencies)
    for file in tqdm.tqdm(python_files, desc='Checking python files'):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('import') or line.startswith('from'):
                    dependency = line.split()[1]
                    # if there is a comma, split it
                    if ',' in dependency:
                        dependency = dependency.split(',')[0]
                    dependencies.add(dependency)
    
    # check dependencies
    missing_dependencies = []
    for dependency in tqdm.tqdm(dependencies, desc='Checking dependencies'):
        # we need ensure the package is from internet, not local, local import should be handled by user
        if dependency.startswith('.'):
            continue
        try:
            __import__(dependency)
        except ImportError:
            missing_dependencies.append(dependency)
        except Exception as e:
            if not quiet:
                loguru.logger.error(f"Error while checking dependency {dependency}: {e}")
            
    # log missing dependencies
    if missing_dependencies:
        loguru.logger.error(f"Missing dependencies: {missing_dependencies}")
        if auto_install:
            loguru.logger.info("Auto installing dependencies...")
            for dependency in missing_dependencies:
                if not quiet:
                    loguru.logger.info(f"Installing {dependency}...")
                pip.main(['install', dependency])
    else:
        loguru.logger.info("All dependencies are satisfied.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check workspace dependencies.')
    parser.add_argument('path', type=str, help='workspace path')
    parser.add_argument('--auto_install', action='store_true', help='auto install dependencies if not found')
    parser.add_argument('--auto_mode', action='store_true', help='auto skip interactive mode')
    parser.add_argument('--quiet', action='store_true', help='quiet mode')
    args = parser.parse_args()
    
    # get running python environment path
    path = os.path.dirname(sys.executable)
    # log python env path & ask for confirmation
    loguru.logger.info(f"Python environment path: {path}")
    if not args.auto_mode:
        if not input("Do you want to continue? (y/n): ").lower() == 'y':
            sys.exit(0)
    
    loguru.logger.info(f"Checking workspace dependencies at {args.path}")
    check_workspace_dependencies(args.path, args.auto_install, args.quiet)
    
    
    
