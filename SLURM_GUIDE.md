# Slurm Guide for DGX Cluster

This guide provides instructions for using the Slurm workload manager on the DGX cluster.

## Initial Setup

### Logging into the Submit Node

Connect to the submit node using SSH:

```bash
ssh wxc321@lvs-radgpujb01
```

### Creating a Screen Session

Create a new screen session to keep your work running even if you disconnect:

```bash
screen -S thread_1
```

**Useful screen commands:**
- Detach from screen: `Ctrl+A` followed by `D`
- Reattach to screen: `screen -r thread_1`
- List screens: `screen -ls`

### Setting Up Your Workspace

Create and navigate to your personal directory on SLURM scratch space:

```bash
mkdir /SLURMSCRATCH/wxc321
cd /SLURMSCRATCH/wxc321
```

## Mounting Network Drives

Initialize Kerberos authentication and mount the group server:

```bash
kinit
sudo mount -t cifs //onfnas01.uwhis.hosp.wisc.edu/radiology/Groups/MIMRTL/Users/Winston/files_to_DGX ./ -o username=wxc321,mfsymlinks,domain=uwhis,uid=wxc321,gid="domain users",sec=krb5i,cruid=wxc321
```

## Starting Interactive Sessions

### Regular Interactive Session

For a basic session with 1 GPU, 4 CPUs, and 16GB memory for 1 hour:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=01:00:00 --pty /bin/bash
```

### QoS-Specific Interactive Sessions

#### Normal QoS (1 day)
```bash
srun --qos=normal --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=1-00:00:00 --pty /bin/bash
```

#### Short QoS (4 hours)
```bash
srun --qos=short --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=04:00:00 --pty /bin/bash
```

#### Long QoS (8 days)
```bash
srun --qos=long --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:2 --mem=32G --time=8-00:00:00 --pty /bin/bash
```

#### High QoS (4 days, requires specific account)
```bash
srun --qos=high --account=dgxgrp2 --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:4 --mem=32G --time=4-00:00:00 --pty /bin/bash
```

## Running Docker Containers

After starting a Slurm session, run a Docker container with MONAI:

```bash
docker run --gpus all -ti -v ./:/local --ipc=host docker.io/projectmonai/monai
```

## Monitoring GPU Usage

Start a separate session for monitoring GPU usage:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=01:00:00 --pty /bin/bash
docker run --gpus all -ti -v ./:/local --ipc=host docker.io/projectmonai/monai
```

**Note:** The system only allows two simultaneous `srun` sessions per user.

## Running Custom Code

Inside your Docker container, run your training and inference code:

```bash
python -m maisi.HU_adapter_run_cv
python -m maisi.HU_adapter_run_inference_cv
```

## Additional Slurm Commands

- Check queue status: `squeue`
- Check job status: `squeue -u wxc321`
- Cancel a job: `scancel [job_id]`
- Check cluster status: `sinfo`

## Tips

1. Always use screen sessions to avoid losing work if your connection drops
2. Mount your data directories before starting Docker
3. Remember to exit your sessions properly when done
4. Monitor your resource usage to optimize your allocation requests 