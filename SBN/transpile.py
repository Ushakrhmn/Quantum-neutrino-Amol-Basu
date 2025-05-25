def produce_transpile_args(config=None, opt_lvl = 3, initial_layout=None, approximation_degree=1.0):
    transpile_args = {
        'optimization_level': opt_lvl
    }

    transpile_args['basis_gates'] = config.supported_instructions if config is not None and hasattr(config, 'supported_instructions') else None
    transpile_args['initial_layout'] = initial_layout
    transpile_args['approximation_degree'] = approximation_degree
        
    return transpile_args