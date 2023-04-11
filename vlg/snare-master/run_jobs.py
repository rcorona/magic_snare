import multiprocessing
import argparse
import pdb
import os
import subprocess
import json
import numpy as np
import csv
import yaml
import itertools
import copy

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
    print('Running: {}'.format(command))
    subprocess.run(command.split())

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
    configs = args.config_names
    exp_names = [cfg.split('/')[-1].split('.')[0] for cfg in configs]

    # Must be same length. 
    assert (len(configs) == len(exp_names))

    # Generate one command per seed. 
    for idx in range(len(configs)):
        commands = commands + ['bash scripts/train.sh {}/{} {}'.format(exp_names[idx], i, configs[idx]) for i in range(args.n_seeds)]

    return commands 

def aggregate_results(args):
    
    # Prep CSV file header.   
    csvfile = open('test.csv', 'w', newline='') 
    writer = csv.writer(csvfile, delimiter=',')
    
    # Seeds for header. 
    seed_h = ['Seed {} ACC'.format(seed) for seed in range(args.n_seeds)]
    
    # Header. 
    header = ['Config Name'] + seed_h
    writer.writerow(header)
    
    # Compute results across all seeds of each config. 
    for cfg_name in args.config_names: 
        
        # Name for experiment folder. 
        name = cfg_name.split('/')[-1]
        exp_dir = 'snap/{}'.format(name)
        
        # Seed directories. 
        seed_nums = list(range(10))
        results = {}
    
        for nums in seed_nums: 
            print(nums)
            results_path = os.path.join(exp_dir, 'checkpoints', 'vl-results-{}.json'.format(nums))
            
            # Read results. 
            result_lines = [r for r in open(results_path, 'r')]
            acc, nonvis, visual = result_lines[1:4]
            
            # Parse
            acc = float(acc.split(':')[-1].split(',')[0].strip())
            nonvis = float(nonvis.split(':')[-1].split(',')[0].strip())
            visual = float(visual.split(':')[-1].split(',')[0].strip())
            
            results['acc'] = results.get('acc', []) + [acc]
            results['nonvis'] = results.get('nonvis', []) + [nonvis]
            results['visual'] = results.get('visual', []) + [visual]
        
        # Print out results. 
        acc = np.mean(results['acc']) * 100.0
        acc_std = np.std(results['acc']) * 100.0
        
        nonvis = np.mean(results['nonvis']) * 100.0
        nonvis_std = np.std(results['nonvis']) * 100.0
        
        visual = np.mean(results['visual']) * 100.0
        visual_std = np.std(results['visual']) * 100.0
        
        print('{}\nACC: {} +/- {}\nBlind: {} +/- {}\nVisual: {} +/- {}\n\n'.format(name, acc, acc_std, nonvis, nonvis_std, visual, visual_std))
            
        # Add row to CSV file. 
        row = [cfg_name] + ['{:2.2f}'.format(r * 100.0) for r in results['acc']]
        
        # Write to file. 
        writer.writerow(row)   
        
    # Close CSV file. 
    csvfile.close() 
    
def make_config_variants(args):
    """
    Generate config files for all variants of a given flag using 
    a provided template config file. All generated configs will be placed
    in the template config's directory. 
    """
    # Load input config file. 
    base_config = yaml.safe_load(open(args.config_variation_file, 'r'))

    # Directory where variant configs should be stored. 
    config_dir = os.path.split(args.config_variation_file)[0]

    # Get Cartesian product of all variable settings to do grid search. 
    flags_settings = [arg.strip().split(',') for arg in args.flags_variant_list]
    product = [p for p in itertools.product(*flags_settings)]

    # Make a config for each variant of parameter provided. 
    for vals in product:
        
        # Will modify copy. 
        config = copy.deepcopy(base_config)
        
        # Replace desired flags with premutation of values. 
        for idx in range(len(vals)):
            
            # Extract pertinent information for flag and setting. 
            f_category, flag = args.variant_flags[idx].split('.')
            val = vals[idx]
            f_type = getattr(__builtins__, args.flag_types[idx])
            
            # Update flag value. 
            config[f_category][flag] = f_type(val)
        
        # Store config variant in configuration dir. 
        config_path = os.path.join(config_dir, '{}.yaml'.format('_'.join(vals)))
        
        with open(config_path, 'w') as f: 
            yaml.dump(config, f)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Batch a set of experiments.')

    # General settings for setting up experiment batches. 
    parser.add_argument('job_type', type=str, default='multiseed')
    parser.add_argument('--gpus', type=str, default='0', help='Comma separated string of GPUs to use.')
    parser.add_argument('--worker_out_dir', type=str, default='.', help='Directory for workers to spit output to.')

    # Arguments for multiseed experiment. 
    #parser.add_argument('--exp_names', type=str, default='test', help='Comma separated names for the parent experiments that are being run across seeds.')
    parser.add_argument('--config_names', type=str, nargs='*', help='Paths to configuration files to use for running experiment.')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds to run experiment with.')

    # Arguments for making variants of configuration files. 
    parser.add_argument('--variant_flags', nargs='*', help='Flag to make variant configs for when tuning.')
    parser.add_argument('--flags_variant_list', nargs='*', help='List of variants for flag to vary when tuning.')
    parser.add_argument('--config_variation_file', type=str, help='Path to directory where to generate all config variations.')
    parser.add_argument('--flag_types', nargs='*', help='Types of each flag being modified.')

    args = parser.parse_args()

    # Generate list of commands to run. 
    if args.job_type == 'multiseed':
        commands = gen_multiseed_commands(args) 
    elif args.job_type == 'aggregate_results':
        aggregate_results(args) 
        exit()  
    elif args.job_type == 'make_config_variants':
        make_config_variants(args)
        exit()

    # Parallelize them to run on GPUs that are available. 
    run_jobs(commands, args.gpus)