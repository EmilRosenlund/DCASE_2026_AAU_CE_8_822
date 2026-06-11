"""
Centralized Configuration Utilities

This module provides shared configuration loading functionality for all pipeline modules.
Instead of duplicating config loading logic in each module, import and use these functions.
"""

import os
from pathlib import Path


def load_runtime_config(config_path=None):
    """
    Load runtime configuration from pipeline/runtime.config
    
    Args:
        config_path: Optional path to config file. If None, auto-detects based on this file's location.
        
    Returns:
        Dictionary with configuration key-value pairs, or None if config not found
    """
    if config_path is None:
        # Auto-detect: go up from utils/ to pipeline/ and find runtime.config
        config_path = Path(__file__).parent.parent / 'runtime.config'
    else:
        config_path = Path(config_path)
    
    config = {}
    
    if not config_path.exists():
        print(f"Warning: runtime.config not found at {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and ':' in line:
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip().strip('"')
    
    return config


def get_environment_paths(config=None, module_type='default'):
    """
    Get data and checkpoint paths based on runtime environment.
    
    Args:
        config: Pre-loaded config dict. If None, loads it automatically.
        module_type: Type of module ('unet', 'tasnet', 'default')
        
    Returns:
        Dictionary with 'data_path', 'checkpoint_dir', 'environment', and module-specific keys
    """
    if config is None:
        config = load_runtime_config()
    
    if config is None:
        # Fallback to local paths if config not found
        script_dir = Path(__file__).resolve()
        project_root = script_dir.parent.parent.parent  # Go up from pipeline/utils to project root
        
        paths = {
            'data_path': str(project_root / 'data'),
            'environment': 'local'
        }
        
        # Add module-specific checkpoint dirs
        if module_type == 'unet':
            paths['checkpoint_dir'] = str(project_root / 'checkpoints' / 'unet')
        elif module_type == 'tasnet':
            paths['checkpoint_dir'] = str(project_root / 'checkpoints' / 'tasnet')
        else:
            paths['checkpoint_dir'] = str(project_root / 'checkpoints')
        
        return paths
    
    environment = config.get('environment', 'local')
    
    # Get data path based on environment
    if environment == 'ailab':
        data_path = config.get('data_path_ailab', '/ceph/project/P8_DCASE/data')
        checkpoint_base = '/ceph/project/P8_DCASE/checkpoints'
    else:
        data_path = config.get('data_path_local', 'C:/Users/emilr/Documents/GitHub/AAU_P8/data')
        checkpoint_base = 'C:/Users/emilr/Documents/GitHub/AAU_P8/checkpoints'
    
    # Build paths dictionary
    paths = {
        'data_path': data_path,
        'environment': environment
    }
    
    # Add module-specific paths
    if module_type == 'unet':
        paths['checkpoint_dir'] = f'{checkpoint_base}/unet'
        paths['output_dir'] = data_path.replace('/data', '/data_unet_separated')
        paths['unet_version'] = config.get('unet_version', 'v1')
    elif module_type == 'tasnet':
        paths['checkpoint_dir'] = f'{checkpoint_base}/tasnet'
        paths['output_dir'] = data_path.replace('/data', '/data_tasnet_separated')
    else:
        paths['checkpoint_dir'] = checkpoint_base
    
    return paths


def get_config_value(key, default=None, config=None):
    """
    Get a single configuration value.
    
    Args:
        key: Configuration key to retrieve
        default: Default value if key not found
        config: Pre-loaded config dict. If None, loads it automatically.
        
    Returns:
        Configuration value or default
    """
    if config is None:
        config = load_runtime_config()
    
    if config is None:
        return default
    
    return config.get(key, default)
