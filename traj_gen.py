import numpy as np

R = 25.4 * 2
H = 25.4 * 1
W = 30

def find_point(t):
    fr = 0.54 * np.pi / 180.0 * (t/1000.) ** (1.)

    y = t
    x = -1*W * np.sin(np.multiply(fr,y))
    z = (R ** 2 - x ** 2) * H / R ** 2

    return np.vstack((x,y,z)).T

def find_normal(p):
    nx = np.zeros((1,len(p[:,0])))

    ny=np.ones(len(p))
    nz=-2*H*p[:,1]/R**2
    nz=-1/nz
    ###nan protection
    ny[np.isinf(nz)] = 0
    nz[np.isinf(nz)] = 1
    ###normalize
    curve_normal=np.vstack((nx,ny,nz)).T
    curve_normal=np.divide(curve_normal,np.tile(np.linalg.norm(curve_normal,axis=1),(3,1)).T)
    idx=np.where(curve_normal[:,-1]>0)
    curve_normal[idx]=-curve_normal[idx]
    return curve_normal

def get_trajectory(x_sample_range):

    curve_p = find_point(x_sample_range)
    curve_n = find_normal(curve_p)
    
    return curve_p, curve_n