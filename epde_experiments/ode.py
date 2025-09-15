import sys
import os
from pathlib import Path
sys.path.append(str(Path().absolute().parent) + '\\EPDE')
sys.path.append(str(Path().absolute().parent))
import time
import numpy as np
import pandas as pd
# import epde.interface.interface as epde
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import traceback
import logging
import os
from epde_integration.hyperparameters import epde_params
from epde_eq_parse.eq_evaluator import evaluate_fronts, EqReranker, FrontReranker
from epde_eq_parse.eq_parser import clean_parsed_out
from epde.interface.interface import EpdeSearch
from epde_integration.hyperparameters import epde_params
import gc

def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * np.std(data) * np.random.normal(size=data.shape) + data

def ode_data():
    base_path = Path().absolute().parent
    path_full = os.path.join(base_path, "data_ode", "ode.npy")
    data = np.load(path_full)
    data = np.transpose(data)
    step = 0.05
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    grids = t
    return data, grids

def ode_discovery(noise_level, epochs):
    data, grid = ode_data()
    dir_name = "ode"
    experiment_info = "noise_" + str(noise_level) + "_epochs_" + str(epochs)
    i = 0
    max_iter_number = 30
    run_eq_info = []

    params = epde_params[dir_name]

    use_solver = params["use_solver"]
    use_pic = params["use_pic"]
    bounds = params["boundary"]

    population_size = params["population_size"]
    training_epochs = epochs

    max_deriv_order = params["max_deriv_order"]
    equation_terms_max_number = params["equation_terms_max_number"]
    data_fun_pow = params["data_fun_pow"]
    additional_tokens = params["additional_tokens"]
    equation_factors_max_number = params["equation_factors_max_number"]
    eq_sparsity_interval = params["eq_sparsity_interval"]

    while i < max_iter_number:
        gc.collect()
        noised_data = noise_data(data, noise_level)
        epde_search_obj = EpdeSearch(use_solver=use_solver, use_pic=use_pic,
                                     boundary=bounds,
                                     coordinate_tensors=grid, device='cuda')

        if noise_level == 0:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                             preprocessor_kwargs={})
        else:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                             preprocessor_kwargs={"use_smoothing": True})
            # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
            #                                  preprocessor_kwargs={'epochs_max' : 1e4})

        epde_search_obj.set_moeadd_params(population_size=population_size,
                                          training_epochs=training_epochs)

        start = time.time()
        epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=max_deriv_order,
                            equation_terms_max_number=equation_terms_max_number, data_fun_pow=data_fun_pow,
                            additional_tokens=additional_tokens,
                            equation_factors_max_number=equation_factors_max_number,
                            eq_sparsity_interval=eq_sparsity_interval) # , data_nn=data_nn
        end = time.time()

        epde_search_obj.equations(only_print=True, num=1)
        res = epde_search_obj.equations(only_print=False, only_str=False, num=1)
        iter_info = evaluate_fronts(res, dir_name, end - start, i)
        front_r = FrontReranker(iter_info)
        run_eq_info.append(front_r.select_best('shd'))
        i += 1
        print(f"Iter #{i}/{max_iter_number} completed")
        print(f"Time spent: {(end-start)/60} min")
    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    eq_r.to_csv(package="epde_experiments", experiment_info=experiment_info)

if __name__ == '__main__':
    ''' Parameters of the experiment '''
    epochs = 5
    noise_level = 0.005
    ''''''
    ode_discovery(noise_level=noise_level, epochs=epochs)




