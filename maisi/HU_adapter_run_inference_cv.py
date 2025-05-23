import os
import subprocess
import time
from datetime import datetime

# Define root directory for consistency
root_dir = "HU_adapter_UNet"

def run_inference_fold(fold, gpu):
    """Run inference for a specific fold on a specific GPU."""
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file for the subprocess stdout/stderr
    log_file = os.path.join(log_dir, f"inference_fold{fold}_gpu{gpu}_process.log")
    command = f"python -m maisi.HU_adapter_inference --fold {fold} --gpu {gpu}"
    
    print(f"Starting inference for fold {fold} on GPU {gpu}")
    print(f"Process log file: {log_file}")
    
    with open(log_file, 'w') as f:
        f.write(f"Started inference for fold {fold} on GPU {gpu} at {datetime.now()}\n")
        f.write(f"Command: {command}\n\n")
        f.flush()
        
        # Create a new environment with the specific GPU assigned
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        # Start the process with the modified environment and redirect output to log file
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env
        )
        
        return process

def main():
    # Create master log file
    log_dir = os.path.join(root_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    master_log = os.path.join(log_dir, f"inference_cross_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    with open(master_log, 'w') as f:
        f.write(f"Inference cross-validation started at {datetime.now()}\n")
        f.write(f"Master log file: {master_log}\n\n")
        
        # Start each fold inference on its own GPU
        processes = []
        for fold in range(1, 5):  # Folds 1 to 4
            gpu = fold + 1  # Map fold to GPU 0-3
            process = run_inference_fold(fold, gpu)
            processes.append((fold, gpu, process))
            
            status_msg = f"Started inference for fold {fold} on GPU {gpu} at {datetime.now()}"
            print(status_msg)
            f.write(status_msg + "\n")
            f.flush()
        
        # Monitor processes
        monitor_msg = "\nAll inference processes started. Monitoring..."
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
                            status_msg = f"Inference for fold {fold} on GPU {gpu} completed successfully at {datetime.now()}!"
                        else:
                            status_msg = f"Inference for fold {fold} on GPU {gpu} failed with return code {process.returncode} at {datetime.now()}"
                        
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
            
            completion_msg = f"\nAll fold inference completed at {datetime.now()}!"
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
                    term_msg = f"Terminating inference for fold {fold} on GPU {gpu}"
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