from __future__ import print_function

import numpy as np

from pymortestbed_lite.ode import dirk1_prim_nl , dirk2_prim_nl , dirk3_prim_nl
from pymortestbed_lite.ode.dirk import DirkPrim
from pymortestbed_lite.linalg        import ScipySpLu as linsolv
from pymortestbed_lite.linalg        import ScipyLstSq as linlstsq
from pymortestbed_lite.optimization.nlsys_intface  import GaussNewton
from pymortestbed_lite.optimization.nlsys_intface  import NewtonRaphson

from pymortestbed_lite.rom.projmr_offline_util import pod

def solv_prim_uns_galerkin(ubar, V, gen, mu, y0, ode, mass, velo, dvelo, t):

    invV = np.linalg.inv(V)

    # Set constant offset
    if ubar is None: ubar = np.zeros(V.shape[0], dtype=float, order='F')

    ## Nonlinear solver
    nlsolv  = NewtonRaphson(linsolv, 25, 1.0e-10, lambda x: np.max(np.abs(x)))
    
    # Number of time steps
    nstep = len(t)-1
    
    # Matrix to store the state at each time step
    y = np.zeros((V.shape[-1], nstep+1), dtype=float, order='F')
    
    # Set first column of matrix to initial state
    y[:, 0] = y0

    # Ensure parameter set appropriately
    msh, phys, disc = gen.freeze_param(mu).freeze_time(np.inf).give_me()

    # Redefine the mass, velo, and dvelo functions
    def nlfunc(y):
        return mass(y, None, msh, phys, disc, 'lmult_transpose', invV), \
               velo(ubar+np.dot(V, y), None, msh, phys, disc, 'lmult', invV), \
               dvelo(y, None, msh, phys, disc, 'lmult', V)

    # Time stepping
    for i, ti in enumerate(t[:-1]):
        # Print progress
        if np.mod(i, len(t)/10) == 0:
            print('>>>>> Timestep {0:d} <<<<<'.format(i))

        # Advance solution one time step
        dt = t[i+1] - ti
        y[..., i+1], Ky = ode(nlfunc, y[..., i], None, ti, dt, gen, nlsolv)

    return y

def solv_prim_uns_LSPG(ubar, V, gen, mu, y0, ode, mass, velo, dvelo, t):
    
    invV = np.linalg.inv(V)

    # Set constant offset
    if ubar is None: ubar = np.zeros(V.shape[0], dtype=float, order='F')

    ## Nonlinear solver
    lmult = lambda A, b: A.transpose().dot(b)
    nlsolv = GaussNewton(linlstsq, 50, 1.0e-8, lambda x: np.max(np.abs(x)),
                         lmult=lmult)
    
    # Extract information and pre-allocate
    # Number of time steps
    nstep = len(t)-1
    # Matrix to store the state at each time step
    yq = np.zeros((V.shape[-1], nstep+1), dtype=float, order='F')
    # Set first column of matrix to initial state
    yq[:, 0] = y0

    # Ensure parameter set appropriately
    msh, phys, disc = gen.freeze_param(mu).freeze_time(np.inf).give_me()

    # Use ode.res() and ode.jac() to define R_{n+1} and its derivative w.r.t. u_{n+1}
    """
    class DirkPrim(Dirk):
        def __init__(self, A, b, c):
            super(DirkPrim, self).__init__(A, b, c)

        def res(self, M, F, KQ, Qm, KQm, Us, dt, msh, phys, disc):
            return dirk_prim_res(self.A, M, F, KQ, Qm, KQm, Us, dt, msh, phys, disc)

        def jac(self, M, dF, KQ, Qm, KQm, Us, dt, msh, phys, disc,
                op='return', *args):
            return dirk_prim_jac(self.A, M, dF, KQ, Qm, KQm, Us, dt,
                                 msh, phys, disc, op, *args)

        Parameters
        ----------
        M, F, dF : functions
            Functions defining ODE (mass matrix, velocity vector, velocity derivative)
        KQ: array-like
            Vectors defining K (stage of integrated quantity) at a particular stage
        Qm : array-like
            Vector defining integrated quantity at previous time (t), i.e. Qm = Q(t)
        KQm : ... x k
            Array of vectors defining stages of integrated quantity at all 'previous'
            stages, i.e. those which have been solved for already
        dt : float
            Timestep
        msh, phys, disc : Mesh, Physics, Discretization class
            Objects defining PDE at time t + ci*dt
    """

    # Time stepping: solve the minimization at each time step
    # pseudo-code: 
    for i, ti in enumerate(t[:-1]):
        # Print progress
        if np.mod(i, len(t)/10) == 0:
            print('>>>>> Timestep {0:d} <<<<<'.format(i))

        # Advance solution one time step
        dt = t[i+1] - ti

        yk = np.zeros((gen.ndof, ode.nstage), dtype=float, order=‘F’)
        # Loop over stages 
        for j in range(ode.nstage):
            def nlfunc(y):
                KQ = np.dot(V, yk[..., j])
                Qm = np.dot(V, yq[..., i])
                KQm = np.dot(V, yk[..., :j])
                return ode.res(mass, velo, KQ, Qm, KQm, None, dt, msh, phys, disc), \
                        ode.jac(mass, dvelo, KQ, Qm, KQm, None, dt, msh, phys, disc, 'rmult', V)

        yq[..., i+1] = nlsolv(nlfunc, yq[..., i])

    return yq

if __name__ == '__main__':

    from pymortestbed_lite.examples.burg_fv_1d.setup import InviscBurgersGenerator
    from pymortestbed_lite.app.burg_fv_1d.invisc import mass, velo, dvelo

    ## Spatial discretization
    lmin, lmax, nel = 0.0, 100.0, 1000
    U0 = np.ones(nel, dtype=float)
    gen = InviscBurgersGenerator(lmin, lmax, nel)

    ## Temporal discretization
    nstage, nstep = 1, 500
    t = np.linspace(0.0, 35.0, nstep+1)
    if   nstage == 1: ode = dirk1_prim_nl
    elif nstage == 2: ode = dirk2_prim_nl
    elif nstage == 3: ode = dirk3_prim_nl

    mu_train = np.array([5.0, 0.02, 0.02])
    U = solv_prim_uns(gen, mu_train, U0, ode, mass, velo, dvelo, t)
    
    # The sensitivity is not required. Just use pod(U)
    # The parameters are different between the steady and unsteady cases 
    # (one is inviscid and the other is viscous Burgers’ equation). 
    # Use the same parameter from the solv_uns.py file for the training and testing (mu = [5.0, 0.02, 0.02]).

    # Q: How to get V/phi? Solve for full model w/ only one training (mu = [5.0, 0.02, 0.02])?
    # Q: phi = qr(pod(U))? 
    # Q: Then run Galerkin/LSPG ROM on testing (mu = [5.0, 0.02, 0.02])? 
    phi, _, _ = pod(U)
    phi, _ = qr(phi, mode='economic')
    
    y0 = np.zeros(phi.shape[-1], dtype=float)

    # Solve inviscid with time dependence 
    # 1. Use Galerkin
    ubar = np.zeros(nel)
    mu_test = np.array([5.0, 0.02, 0.02])
    U = solv_prim_uns_Galerkin(ubar, phi, gen, mu_test, y0, ode, mass, velo, dvelo, t)
    # 2. Use LSPG
    # ode = ... -> use DirkPrim (need to define it in __init__.py under pymortestbed_lite.ode) 
    # U = solv_prim_uns_LSPG(ubar, phi, gen, mu_test, y0, ode, mass, velo, dvelo, t)

    # Postprocess
    msh, _, _ = gen.give_me()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for i in np.linspace(0, nstep, 10, dtype=int):
        plt.plot(msh.get_dual_nodes(), U[:, i], lw=2)
    plt.show()