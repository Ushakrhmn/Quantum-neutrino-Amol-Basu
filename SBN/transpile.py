import json

def produce_transpile_args(config=None, path = None):
    with open(path, 'r') as f:
        transpile_config = json.load(f)

    transpile_args = {
        'optimization_level': transpile_config.get('optimization_level', 3),
    }

    transpile_args['basis_gates'] = config.supported_instructions if config is not None and hasattr(config, 'supported_instructions') else None
    transpile_args['initial_layout'] = transpile_config.get('initial_layout', None)
    transpile_args['approximation_degree'] = transpile_config.get('approximation_degree', 1)
        
    return transpile_args