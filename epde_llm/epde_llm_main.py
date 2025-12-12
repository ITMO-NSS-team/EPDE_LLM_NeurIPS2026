import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent) + '\\EPDE')
sys.path.append(str(Path().absolute().parent))
from pipeline.optimization_workflow.optimization_manager import OptManager
from epde_integration.epde_search import EpdeSearcher
from epde_integration import hyperparameters
from epde_eq_parse.eq_evaluator import EqReranker
import numpy as np
import time


llm_iter_num = 0
max_iter = 6
start_iter = 0
refine_point = 100
epde_llm_iterations = 30

debug = False # True False
print_exc = True
exit_code = False

data_args = {"resample_shape": (1, 1),
             "use_cached": False,
             "noise_level": 0,
             "dir_name": "burg"}



# проверить что data матрицы совпадают и их не надо .T
if __name__ == '__main__':
    dir_name = data_args["dir_name"]
    experiment_info = "noise_" + str(data_args["noise_level"]) + "_epochs_" + str(hyperparameters.epde_params[data_args["dir_name"]]["training_epochs"])
    run_eq_info = []
    for epde_llm_iteration in range(epde_llm_iterations):
        t1 = time.time()
        opt_manager = OptManager(max_iter, start_iter, refine_point, debug, print_exc, exit_code,
                                     data_args, n_candidates=4, llm_iter=llm_iter_num)
        opt_manager.explore_solutions()

        pruned_track, not_pruned = opt_manager.call_pruner()
        full_records_track = opt_manager.eq_buffer.full_records_track
        data = opt_manager.evaluator.data['inputs'] # "inputs": [raw_data['t'], raw_data['x'], raw_data['u']]
        epde_searcher = EpdeSearcher(data, full_records_track, pruned_track, data_args['dir_name'], use_init_population=True,
                                        max_iter_num=1, device='cuda', noise_level=data_args["noise_level"], start=t1)
        run_eq_info.append(epde_searcher.fit())
        t2 = time.time()
        print(f"Iter #{epde_llm_iteration + 1}/{epde_llm_iterations} completed")
        print(f"Time spent: {(t2 - t1) / 60} min")
    eq_r = EqReranker(run_eq_info, dir_name)
    eq_r.best_run_inf = run_eq_info
    # best_info = eq_r.select_best('shd')
    eq_r.to_csv(package="epde_llm", experiment_info=experiment_info)

    # 'd^2u/dt^2 = c[0] * du/dx + c[1]*x + c[2]'
    # 'd^2u/dt^2 = .000423725096 * du/dx + 0.0170837104*x + -1.05863540'
    # -1.0586354035340615 * du/dx1 + 0.0004237250964155818 * x + 0.017083710392065486
    # {'sparsity': {'optimizable': True, 'value': 1.0}, 'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': 1}}
