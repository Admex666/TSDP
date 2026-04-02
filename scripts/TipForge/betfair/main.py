import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - STARTING - %(message)s')

def run_script(script_name):
    logging.info(f"Running {script_name}...")
    try:
        # Use python to run the script
        result = subprocess.run(["python", script_name], check=True)
        logging.info(f"Finished {script_name} successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e}")
        raise

def main():
    scripts = [
        "02_clean_and_features.py",
        "03_backtest.py",
        "04_report.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            logging.error(f"Script {script} not found in current directory.")

if __name__ == "__main__":
    main()
