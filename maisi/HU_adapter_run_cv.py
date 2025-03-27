import os
import subprocess
import time
from datetime import datetime

def run_cv_fold(fold, gpu):
    """Run cross-validation training for a specific fold on a specific GPU."""
    log_dir = os.path.join("HU_adapter_UNet", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Two log files - one for the subprocess stdout/stderr and one for the detailed training logs
    # The detailed training logs are handled by the training script itself
    log_file = os.path.join(log_dir, f"fold{fold}_gpu{gpu}_process.log")
    command = f"python -m maisi.HU_adapter_train_cv --fold {fold} --gpu {gpu}"
    
    print(f"Starting fold {fold} on GPU {gpu}")
    print(f"Process log file: {log_file}")
    print(f"Detailed output will be saved to: {os.path.join(log_dir, f'fold{fold}_gpu{gpu}_detailed.log')}")
    
    with open(log_file, 'w') as f:
        f.write(f"Started training fold {fold} on GPU {gpu} at {datetime.now()}\n")
        f.write(f"Command: {command}\n\n")
        f.flush()
        
        # Start the process and redirect output to log file
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        
        return process

def main():
    # Create master log file
    log_dir = os.path.join("HU_adapter_UNet", "logs")
    os.makedirs(log_dir, exist_ok=True)
    master_log = os.path.join(log_dir, f"cross_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    with open(master_log, 'w') as f:
        f.write(f"Cross-validation started at {datetime.now()}\n")
        f.write(f"Master log file: {master_log}\n\n")
        
        # First ensure we have the fold data
        if not os.path.exists(os.path.join("HU_adapter_UNet", "folds.json")):
            print("Creating folds first...")
            f.write("Creating folds first...\n")
            f.flush()
            result = os.system("python -m maisi.HU_adapter_create_folds")
            if result != 0:
                error_msg = f"Error creating folds. Exiting with code {result}"
                print(error_msg)
                f.write(error_msg + "\n")
                return
        
        # Start each fold on its own GPU
        processes = []
        for fold in range(1, 5):  # Folds 1 to 4
            gpu = fold - 1  # Map fold to GPU 0-3
            process = run_cv_fold(fold, gpu)
            processes.append((fold, gpu, process))
            
            status_msg = f"Started fold {fold} on GPU {gpu} at {datetime.now()}"
            print(status_msg)
            f.write(status_msg + "\n")
            f.flush()
        
        # Monitor processes
        monitor_msg = "\nAll processes started. Monitoring..."
        print(monitor_msg)
        f.write(monitor_msg + "\n")
        f.flush()
        
        try:
            while processes:
                # Create a list to store processes that have completed
                completed = []
                
                # Check each process
                for fold, gpu, process in processes:
                    if process.poll() is not None:  # Process completed
                        if process.returncode == 0:
                            status_msg = f"Fold {fold} on GPU {gpu} completed successfully at {datetime.now()}!"
                        else:
                            status_msg = f"Fold {fold} on GPU {gpu} failed with return code {process.returncode} at {datetime.now()}"
                        
                        print(status_msg)
                        f.write(status_msg + "\n")
                        f.flush()
                        
                        # Add to completed list instead of modifying the original list
                        completed.append((fold, gpu, process))
                
                # Remove completed processes from the original list
                for proc in completed:
                    if proc in processes:
                        processes.remove(proc)
                
                if processes:
                    time.sleep(10)  # Check every 10 seconds
            
            completion_msg = f"\nAll fold training completed at {datetime.now()}!"
            print(completion_msg)
            f.write(completion_msg + "\n")
            f.flush()
        
        except KeyboardInterrupt:
            interrupt_msg = f"\nInterrupted at {datetime.now()}! Terminating all processes..."
            print(interrupt_msg)
            f.write(interrupt_msg + "\n")
            f.flush()
            
            for fold, gpu, process in processes:
                if process.poll() is None:  # Process still running
                    term_msg = f"Terminating fold {fold} on GPU {gpu}"
                    print(term_msg)
                    f.write(term_msg + "\n")
                    f.flush()
                    process.terminate()
            
            terminated_msg = "All processes terminated."
            print(terminated_msg)
            f.write(terminated_msg + "\n")
            f.flush()

if __name__ == "__main__":
    main() 