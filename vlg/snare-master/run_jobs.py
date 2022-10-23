import multiprocessing
import argparse
import pdb
import os
import subprocess

def permute_combos(list1, list2, l1_prefix='', l2_prefix=''):
    """
    For every item in list1, create a pairing with every item in list 2
    and consolidate into single list. 
    """
    pairs = []

    for e1 in list1: 
        for e2 in list2: 
            pairs.append(' '.join((l1_prefix + e1, l2_prefix + e2 + ' ')))

    return pairs

def get_arg(cmd_list, arg):
    """
    Given list of command line arguments, extract the value of argument arg. 
    """
    return cmd_list[cmd_list.index(arg) + 1]

def run_job(command_args):
    
    # Unpack. 
    command, args = command_args
    gpus = args.gpus.split(',')

    # Determine which GPU to run on based off of process ID. 
    pid = multiprocessing.current_process()._identity[0] - 1

    # Set up usage of desired GPU. 
    gpu_id = gpus[pid]
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Call command with redirected output to not confuse command line. 
    out_path = os.path.join(args.worker_out_dir, '{}_out.txt'.format(pid))
    subprocess.run(command.split(), stdout=open(out_path, 'w'))

def run_jobs(commands, gpus):
    """
    Given a list of commands and GPUs, parallelize their execution. 
    """
    # Unpack GPU list. 
    gpus = gpus.split(',')

    """
    for i in range(len(gpus)):
        print('CUDA_VISIBLE_DEVICES={} {} &'.format(gpus[i], commands[i]))
    """

    # CUDA doesn't support fork. 
    multiprocessing.set_start_method('spawn')

    # Run all the jobs. 
    with multiprocessing.Pool(len(gpus)) as p: 
        
        # Send GPUs along with commands to workers. 
        commands = [[command, args] for command in commands]
        
        p.map(run_job, commands)

def gen_multiseed_commands(args):
    """
    Generate commands to run the same experiment over multiple seeds spread across GPUs. 
    """

    # Accumulate commands across config files. 
    commands = []

    # Get all experiment configs and names. 
    configs = args.config_names.split(',')
    exp_names = args.exp_names.split(',')

    # Must be same length. 
    assert (len(configs) == len(exp_names))

    # Generate one command per seed. 
    for idx in range(len(configs)):
        commands = commands + ['bash scripts/train.sh {}_{} {}'.format(exp_names[idx], i, configs[idx]) for i in range(args.n_seeds)]

    return commands 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Batch a set of experiments.')

    # General settings for setting up experiment batches. 
    parser.add_argument('job_type', type=str, default='multiseed')
    parser.add_argument('--gpus', type=str, default='0', help='Comma separated string of GPUs to use.')
    parser.add_argument('--worker_out_dir', type=str, default='.', help='Directory for workers to spit output to.')

    # Arguments for multiseed experiment. 
    parser.add_argument('--exp_names', type=str, default='test', help='Comma separated names for the parent experiments that are being run across seeds.')
    parser.add_argument('--config_names', type=str, help='Comma separated paths to configuration files to use for running experiment.')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds to run experiment with.')

    args = parser.parse_args()

    # Generate list of commands to run. 
    if args.job_type == 'multiseed':
        commands = gen_multiseed_commands(args)    

    # Parallelize them to run on GPUs that are available. 
    run_jobs(commands, args.gpus)