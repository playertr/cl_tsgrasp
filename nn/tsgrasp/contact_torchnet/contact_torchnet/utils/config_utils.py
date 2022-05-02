# config_utils.py
# Tim Player, playert@oregonstate.edu, October 13 2021
# Lightly modified from Contact-Graspnet

import yaml

def load_config(config_path, arg_configs=[]):
    """
    Loads yaml config file and overwrites parameters with function arguments and --arg_config parameters

    Arguments:
        config_path {str} -- path to .yaml file

    Keyword Arguments:
        arg_configs {list} -- Overwrite config parameters by hierarchical command line arguments (default: {[]})

    Returns:
        [dict] -- Config
    """

    with open(config_path,'r') as f:
        global_config = yaml.load(f)

    for conf in arg_configs:
        k_str, v = conf.split(':')
        try:
            v = eval(v)
        except:
            pass
        ks = [int(k) if k.isdigit() else k for k in k_str.split('.')]
        
        recursive_key_value_assign(global_config, ks, v)

    return global_config

def recursive_key_value_assign(d,ks,v):
    """
    Recursive value assignment to a nested dict

    Arguments:
        d {dict} -- dict
        ks {list} -- list of hierarchical keys
        v {value} -- value to assign
    """
    
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]],ks[1:],v)
    elif len(ks) == 1:
        d[ks[0]] = v