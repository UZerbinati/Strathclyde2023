from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from netgen.geom2d import SplineGeometry
from pickle import dump,load
from scipy.linalg import eigvals,eigvalsh
from scipy.io import savemat
import numpy as np

NDoFs = [2**3]
NEig = 10
Err = []
Which =0

Exact = [13.086172791,23.031098494,23.031098494,32.052396078]
for N in NDoFs:
    msh = RectangleMesh(N, N, 1, 1,originX=-1.,originY=-1.,diagonal="crossed")
    V = VectorFunctionSpace(msh, "CG", 4) 
    Q = FunctionSpace(msh, "DG", 3)
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
            File("VTK/P4P3Disc.pvd").write(u)
    print("Errors: ", Err)
    fp = open('Errors/ErrorP4P3Disc.pkl', 'wb')
    dump(Err,fp)