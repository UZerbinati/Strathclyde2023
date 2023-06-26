#-st_ksp_pc_side right   -st_pc_type lu -st_pc_factor_mat_solver_type mumps -st_mat_mumps_icntl_14 100 -st_mat_mumps_icntl_24 1 -st_mat_mumps_cntl_5 1.e14 -st_type sinvert
import matplotlib.pyplot as plt
from pickle import dump,load
import numpy as np

plt.figure()
fp = open("Errors/ErrorP2P1Disc.pkl",'rb')
data = load(fp)
Ds = [sol[0] for sol in data]
Errs = [sol[2] for sol in data] 
plt.loglog(Ds,Errs,"*-")
for i in range(len(data)):
    print("[{}] P2-P1Disc: {}".format(i,np.polyfit(np.log(Ds[i:i+2]), np.log(Errs[i:i+2]),1)))
fp = open("Errors/ErrorP2P1.pkl",'rb')
data = load(fp)
Ds = [sol[0] for sol in data]
Errs = [sol[2] for sol in data] 
plt.loglog(Ds,Errs,"*-")
for i in range(len(data)):
    print("[{}] P2-P1: {}".format(i,np.polyfit(np.log(Ds[i:i+2]), np.log(Errs[i:i+2]),1)))
fp = open("Errors/ErrorP2P2.pkl",'rb')
data = load(fp)
Ds = [sol[0] for sol in data]
Errs = [sol[2] for sol in data] 
plt.loglog(Ds,Errs,"*-")
for i in range(len(data)):
    print("[{}] P2-P2: {}".format(i,np.polyfit(np.log(Ds[i:i+2]), np.log(Errs[i:i+2]),1)))
plt.loglog(Ds[1:],[20*d**2 for d in Ds[1:]],"--")
plt.loglog(Ds[1:],[30*d**4 for d in Ds[1:]],"--")
plt.legend(["P2-P1Disc","P2-P1","P2-P2","2nd Order","4th Order"])
plt.ylabel(r"$|\lambda_1-\lambda^h_1|$")
plt.xlabel(r"$h$")
plt.show()