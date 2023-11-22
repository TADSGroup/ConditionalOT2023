import ot
import numpy as np

def pointset_pushforward_ot(reference,target,cost_matrix,reg=0.1,solver='lp'):
    '''
    estimates a monge solution by averaging over the Kantorovich solution
    Note that if len(reference)=len(target), and we are using LP, then this gives a true monge solution

    reference should be a 2d array, with each row being a sample
    target should be a 2d array, with each row being a sample
    '''
    n=len(reference)
    m=len(target)
    if solver=='lp':
        ot_sol=ot.lp.emd(np.ones(n), n*np.ones(m)/m, cost_matrix,numItermax=1e6)
    elif solver=='sinkhorn':
        ot_sol=ot.sinkhorn(np.ones(n), n*np.ones(m)/m, cost_matrix,reg=reg)
    else:
        raise Exception("solver must be one of 'lp' or 'sinkhorn'")
    return ot_sol@target

def get_epsilon_quadratic_cost_mat(reference,target,eps,dim_shared):
    return (
        ot.dist(reference[:,:dim_shared],target[:,:dim_shared])/eps+
        ot.dist(reference[:,dim_shared:],target[:,dim_shared:])
    )

def build_regression_problem(
    reference,
    target,
    conditioning_eps,
    dim_shared,
    solver = 'lp',
    sinkhorn_reg = 0.05
    ):
    print("Building Cost Matrix")
    cost_mat=get_epsilon_quadratic_cost_mat(reference,target,eps=conditioning_eps,dim_shared=dim_shared)

    print("Solving OT Problem")
    mapped_points = pointset_pushforward_ot(reference,target,cost_mat,solver=solver,reg = sinkhorn_reg)
    reg_y=mapped_points[:,dim_shared:]
    reg_X=reference
    print("Finished")
    return reg_X,reg_y