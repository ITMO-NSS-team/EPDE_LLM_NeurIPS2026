class WaveSch(object):
    """
    Static class to store wave equation schema and parameters.
    """
    left_side = 'd^2u/dx0^2'
    schema = frozenset({'d^2u/dx0^2', 'd^2u/dx1^2', 'C'})
    params = {'d^2u/dx0^2': -1.0, 'd^2u/dx1^2': 0.04, 'C': 0.0}


class OdeSch(object):
    """
    Static class to store wave equation schema and parameters.
    """
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u', 'x_0', 'd^2u/dx0^2', 'C'})
    params = {'du/dx0': -1.0, 'u': 4.0, 'x_0': 1.5, 'd^2u/dx0^2': -1.0, 'C': 0.0}


class BurgSch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'C'})
    params = {'du/dx0': 1.0, 'u * du/dx1': 1.0, 'C': 0.0}


class BurgSindySch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^2u/dx1^2', 'C'})
    params = {'du/dx0': -1.0, 'u * du/dx1': -1.0, 'd^2u/dx1^2': -0.1, 'C': 0.0}
    # correct_params5 = {'du/dx0': -1.0, 'u * du/dx1': 1.0, 'd^2u/dx1^2': 0.1, 'C': 0.0}
    # correct_params6 = {'du/dx0': -10.0, 'u * du/dx1': 10.0, 'd^2u/dx1^2': 1.0, 'C': 0.0}


class KdvSindySch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^3u/dx1^3', 'C'})
    params = {'du/dx0': -1.0, 'u * du/dx1': -6.0, 'd^3u/dx1^3': -1.0, 'C': 0.0}


class KdvSch(object):
    left_side = 'du/dx0'
    schema = frozenset({'du/dx0', 'u * du/dx1', 'd^3u/dx1^3', 'cos(t)sin(x)', 'C'})
    params = {'du/dx0': 1.0, 'u * du/dx1': 6.0, 'd^3u/dx1^3': 1.0, 'cos(t)sin(x)': -1.0, 'C': 0.0}


schemas = {'wave': {'schema': WaveSch.schema,
                    'params': WaveSch.params,
                    'left_side': WaveSch.left_side,},

           'ode': {'schema': OdeSch.schema,
                    'params': OdeSch.params,
                    'left_side': OdeSch.left_side,},

           'burg_sindy': {'schema': BurgSindySch.schema,
                          'params': BurgSindySch.params,
                          'left_side': BurgSindySch.left_side,},

           'burg': {'schema': BurgSch.schema,
                    'params': BurgSch.params,
                    'left_side': BurgSch.left_side,},

           'kdv': {'schema': KdvSch.schema,
                   'params': KdvSch.params,
                   'left_side': KdvSch.left_side,},

           'kdv_sindy': {'schema': KdvSindySch.schema,
                         'params': KdvSindySch.params,
                         'left_side': KdvSindySch.left_side,},
           }
