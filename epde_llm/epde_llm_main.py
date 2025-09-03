import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent) + '\\EPDE')
sys.path.append(str(Path().absolute().parent))
from pipeline.optimization_workflow.optimization_manager import OptManager
from epde_integration.epde_search import EpdeSearcher
import numpy as np

llm_iter_num = 0
max_iter = 6
start_iter = 0
refine_point = 100

debug = False # True False
print_exc = True
exit_code = True

data_args = {"resample_shape": (20, 20),
             "use_cached": False,
             "noise_level": 0,
             "dir_name": "wave"}


# проверить что data матрицы совпадают и их не надо .T
if __name__ == '__main__':
    opt_manager = OptManager(max_iter, start_iter, refine_point, debug, print_exc, exit_code,
                                 data_args, n_candidates=4, llm_iter=llm_iter_num)
    opt_manager.explore_solutions()

    pruned_track, not_pruned = opt_manager.call_pruner()
    full_records_track = opt_manager.eq_buffer.full_records_track
    data = opt_manager.evaluator.data['inputs'] # "inputs": [raw_data['t'], raw_data['x'], raw_data['u']]
    epde_searcher = EpdeSearcher(data, full_records_track, pruned_track, data_args['dir_name'], use_init_population=True,
                                                                                   max_iter_num=1, device='cuda')
    run_eq_info = epde_searcher.fit()
    print()

    # 'd^2u/dt^2 = c[0] * du/dx + c[1]*x + c[2]'
    # 'd^2u/dt^2 = .000423725096 * du/dx + 0.0170837104*x + -1.05863540'
    # -1.0586354035340615 * du/dx1 + 0.0004237250964155818 * x + 0.017083710392065486
    # {'sparsity': {'optimizable': True, 'value': 1.0}, 'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': 1}}
