from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens
import numpy as np


custom_trigonometric_eval_fun = {
    'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                        eval_fun_params_labels=['power'])
trig_params_ranges = {'power': (1, 1)}
trig_params_equal_ranges = {}

custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                  token_labels=['cos(t)sin(x)'],
                                  evaluator=custom_trig_evaluator,
                                  params_ranges=trig_params_ranges,
                                  params_equality_ranges=trig_params_equal_ranges,
                                  meaningful=True, unique_token_type=False)

epde_params = {
    'burg': {'boundary': (20, 20),
             'population_size': 8,
             'training_epochs': 5,
             'max_deriv_order': (2, 3),
             'equation_terms_max_number': 5,
             'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.9, 0.1]},
             'eq_sparsity_interval': (1e-6, 1e-5),
             'num': 1,
             'additional_tokens': None,
             "use_solver": False,
             "use_pic": True,
             "data_fun_pow": 3,
             "fourier_layers": True},

    'burg_sindy': {'boundary': (20, 50),
             'population_size': 8,
             'training_epochs': 5,
             'max_deriv_order': (2, 3),
             'equation_terms_max_number': 5,
             'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.9, 0.1]},
             'eq_sparsity_interval': (1e-6, 1e-5),
             'num': 1,
             'additional_tokens': None,
             "use_solver": False,
             "use_pic": True,
             "data_fun_pow": 3,
             "fourier_layers": True},

    'kdv': {'boundary': (20, 20),
             'population_size': 8,
             'training_epochs': 5,
             'max_deriv_order': (2, 3),
             'equation_terms_max_number': 5,
             'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.9, 0.1]},
             'eq_sparsity_interval': (1e-6, 1e-5),
             'num': 1,
             'additional_tokens': None,
             "use_solver": False,
             "use_pic": True,
             "data_fun_pow": 3,
             "fourier_layers": True},

    'kdv_sindy': {'boundary': (40, 100),
             'population_size': 8,
             'training_epochs': 5,
             'max_deriv_order': (2, 3),
             'equation_terms_max_number': 5,
             'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.9, 0.1]},
             'eq_sparsity_interval': (1e-6, 1e-5),
             'num': 1,
             'additional_tokens': None,
             "use_solver": False,
             "use_pic": True,
             "data_fun_pow": 3,
             "fourier_layers": True},

    'wave': {'boundary': (20, 20),
             'population_size': 8,
             'training_epochs': 5,
             'max_deriv_order': (2, 3),
             'equation_terms_max_number': 5,
             'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.9, 0.1]},
             'eq_sparsity_interval': (1e-6, 1e-5),
             'num': 1,
             'additional_tokens': None,
             "use_solver": False,
             "use_pic": True,
             "data_fun_pow": 3,
             "fourier_layers": True},
}