import numpy as np
import itertools
from . import cohortshapley as cs

def legal_permutations_generator(union_structure):
    #union_structure must be a list of lists of integers where each sublist represents one union
    for outer_perm in itertools.permutations(union_structure):
        for v in itertools.product(*map(lambda v: list(itertools.permutations(v)),outer_perm)):
            yield(tuple(itertools.chain(*v)))

class CohortOwen(cs.CohortShapley):
    def __init__(self, union_structure, model, similarity, subject_id, data, func=np.average, y=None,
            parallel=0, pid=0, data_weight=None, permutations=None,
            mc_num = None,
            verbose=1):
        super().__init__(model=model, similarity=similarity, subject_id=subject_id, data=data, func=func, y=y,
            parallel=parallel, pid=pid, data_weight=data_weight, permutations=permutations, mc_num=mc_num, verbose=verbose)
        self.union_structure = union_structure
        if  mc_num != None:
            n_vars = data.shape[-1]
            n_unions = len(union_structure)
            permutations = np.zeros((mc_num, n_vars), dtype=int)
            for k in range(mc_num):
                union_order = np.random.permutation(n_unions)
                perm = []
                for union_ind in union_order:
                    union = union_structure[union_ind]
                    perm = np.append(perm,np.random.permutation(union))
                permutations[k] = perm
            self.permutations = permutations
        else:
            self.permutations = np.array(list(legal_permutations_generator(union_structure)))

            
