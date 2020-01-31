import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dt = .01
num_steps = int(400)
ens_num = 50
obs_freq =  5 # every X*dt seconds


def lorenz(u,v,w, sigma=10., beta=2.667, rho=28.):
    """The Lorenz equations."""
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def ensemble(h=.001, ens_num = ens_num, dt=dt, num_steps=num_steps):
    u_ensemble = np.empty((ens_num, num_steps))
    w_ensemble = np.empty((ens_num, num_steps))
    v_ensemble = np.empty((ens_num, num_steps))
     
    # run for each ens
    for k in range(ens_num):
        u0 = 0.0 + np.random.randn()*h
        v0 = 1.0 + np.random.randn()*h 
        w0 = 1.05 + np.random.randn()*h
        
        # set up stuff 
        us = np.empty(num_steps)
        vs = np.empty(num_steps)
        ws = np.empty(num_steps)
        
        #
        us[0] = u0
        vs[0] = v0
        ws[0] = w0
        # loop through time  
        for i in range(1, num_steps):
            du, dv, dw = lorenz(us[i-1], vs[i-1], ws[i-1])
            us[i] = us[i-1] + du*dt 
            vs[i] = vs[i-1] + dv*dt 
            ws[i] = ws[i-1] + dw*dt 
        #member = np.vstack([us, vs, ws])
        #ensemble[k*3:(k+1)*3, :] = member 
        u_ensemble[k] = us
        v_ensemble[k] = vs
        w_ensemble[k] = ws
    
    # return things 
    return u_ensemble, v_ensemble, w_ensemble


# Nonlinear Forward Model
def G(psi, dt=dt, func=lorenz):
    # psi is a list or an array of dim 3 
    u,v,w = psi[0], psi[1], psi[2]
    du, dv, dw = func(u,v,w)
    u = u + du*dt 
    v = v + dv*dt 
    w = w + dw*dt 
    return np.array([u,v,w])

# Model error   # this is only used to run the 'true' model
## covariance 
C_qq = [[.1,0,0],     
        [0,.1,0],
        [0,0,.1]]
## mean 
q_mn = [0,0,0]


## Measurements 
# Measurement model
M = np.eye(3,3)                                   

# Error covariance matrix 
C_ee = [[.001 ,0, 0],    # 
        [0 ,.001, 0],
        [0, 0, .001]]
# Mean
e_mn = [0,0,0]

# Identity matrix 
I = np.eye(3,3)

# Ensemble of forecasts. Use to contruct forecast covariance model 
ue, ve, we = ensemble()

# Truth run
uT, vT, wT = ensemble(h=0, ens_num=1)  # 'truth' run
psi_T = np.vstack([uT, vT, wT]).T


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
# !!! Run the Forward Model and Perform Filtering !!!  
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

# set the initial state
psi_a = np.array([0,1.,1.05])

# collect the outputs 
psi_D = np.empty((num_steps,3))   # the measurement vector 
psi_F = np.empty((num_steps, 3))  # the 'forecast' vector
psi_A = np.empty((num_steps, 3))  # the 'analysis' vector 

#set the initial conditions in the output vectors 
psi_D[0,:] = psi_a
psi_F[0,:] = psi_a
psi_A[0,:] = psi_a


# figure  
fig = plt.figure()
ax = fig.gca(projection='3d')


# run forward through time 
for k in range(1,num_steps):
    
    # Random multivariate normal  
    q = np.random.multivariate_normal(q_mn, C_qq)
    
    # Run the forward model
    psi_f = G(psi_a) + q 
    
    # Compute the forecast error covariance 
    #C_ff = G * C_aa * G.T + C_qq --- for the case of the normal kalman filter 
    C_ff_k = np.cov([ue[:,k], ve[:,k], we[:,k]])

    # compute the Kalman gain 
    K = C_ff_k*M.T*np.linalg.inv((M*C_ff_k*M.T + C_ee))
    
    # Compute the analysis covariance matrix 
    C_aa = (I - K*M)*C_ff_k    # we do this even when we don't have a datapoint!

    # set the 'analysis' state to the forecast-- this gets overwritten if data comes in
    psi_a = psi_f 
    
    # data is assigned a nan value for each time--unless it gets observed below
    d = np.nan

    # perform the filtering --- observations are made 
    if k%obs_freq == 0:   # if the timestep is perfectly divisible by 10
        
        # perfom an observation
        e = np.random.multivariate_normal(e_mn, C_ee)
        d = M.dot(psi_T[k,:]) + e # + the meaasurement noise??
        
        # compute the analysis 
        psi_a = psi_f + K.dot(d - M.dot(psi_f))
    
    # assign things to the output arrays 
    psi_A[k,:] = psi_a
    psi_F[k,:] = psi_f
    psi_D[k,:] = d
    
# plot 
ax.plot(psi_F[:,0], psi_F[:,1], psi_F[:,2], color='blue', label='forecast')
ax.plot(psi_A[:,0], psi_A[:,1], psi_A[:,2], color='black', label='analysis')
ax.plot(psi_T[:,0], psi_T[:,1], psi_T[:,2], color='red', label='truth')
ax.scatter(psi_D[:,0], psi_D[:,1], psi_D[:,2], color='black', marker='x', label='observations')
ax.scatter(psi_A[0,0], psi_A[0,1], psi_A[0,2], label = 'starting point')
plt.show()
plt.legend()
