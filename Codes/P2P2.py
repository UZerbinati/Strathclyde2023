from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from netgen.geom2d import SplineGeometry
from pickle import dump,load
from scipy.linalg import eigvals,eigvalsh
from scipy.io import savemat
import numpy as np

NDoFs = [2**2,2**3,2**4,2**5,2**6,2**7]
NEig = 10
Err = []
Which =0

Exact = [13.086172791,23.031098494,23.031098494,32.052396078]
for N in NDoFs:
    msh = RectangleMesh(N, N, 1, 1,originX=-1.,originY=-1.,diagonal="left")
    V = VectorFunctionSpace(msh, "CG", 2) 
    Q = FunctionSpace(msh, "CG", 2)
    X = V*Q
    print("Number of DoFs {}".format(sum(X.dof_count)))
    u,p = TrialFunctions(X)
    v,q = TestFunctions(X)

    nu = 1;

    a = nu*(inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q))*dx
    m = inner(u,v)*dx
    
    bc = DirichletBC(X.sub(0), as_vector([0.0,0.0]) , [1,2,3,4]) # Boundary condition
    print("Problem Set Up !")

    sol = Function(X)

    print("Assembling ...")
    A = assemble (a,bcs=bc)# There is an apperent difference in imposing the boudnary
    M = assemble (m,bcs=bc,weight=0.)# condition using DirichletBC or a penalty method.
    Asc, Msc = A.M.handle, M.M.handle
    E = SLEPc.EPS().create()
    E.setType(SLEPc.EPS.Type.ARNOLDI)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP);
    E.setDimensions(NEig,SLEPc.DECIDE);
    E.setOperators(Asc,Msc)
    E.setMonitor(lambda eps,it,nconv,lr,li: print("[{}] Number of Converged Eig {}".format(it,nconv)))
    E.setFromOptions()
    E.solve()
    nconv = E.getConverged()
    print("Number of converged eigenvalues is: {}".format(nconv))
    for i in range(nconv): 
        vr, vi = Asc.getVecs()
        with sol.dat.vec_wo as vr:
            lam = E.getEigenpair(i, vr, vi)
        u,p = sol.split()
        u.rename("Velocity")
        print("[{}] Eigenvalue: {} -- Div: {}".format(i,lam.real,sqrt(assemble(dot(div(u),div(u)) * dx))))
        if i == Which:
            Err = Err + [(1/N,sum(X.dof_count),np.abs(lam.real-Exact[Which]))]
            File("VTK/P2P2.pvd").write(u)
    print("Errors: ", Err)
    fp = open('Errors/ErrorP2P2.pkl', 'wb')
    dump(Err,fp)

import matplotlib.pyplot as plt
fp = open("Errors/ErrorP2P2.pkl",'rb')
data = load(fp)
plt.figure()
plt.title(r"$|\lambda_1-\lambda^h_1|$")
Ds = [sol[0] for sol in data]
Errs = [sol[2] for sol in data] 
plt.loglog(Ds,Errs,"*-")
plt.loglog(Ds,[30*d**2 for d in Ds],"--")
for i in range(len(data)):
    print("[{}] P2-P2: {}".format(i,np.polyfit(np.log(Ds[i:i+2]), np.log(Errs[i:i+2]),1)))
plt.legend(["P2-P2","2nd Order"])
plt.show()


