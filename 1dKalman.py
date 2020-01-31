import numpy as np
from matplotlib import pyplot as plt 


# model parameters 
vk = 1.0 #m/s
dt = .1  #s
tlen = 200 # model time steps 

# Measurement and Forward Model
M = np.eye(2,2)                                   # identify matrix
G = np.array([[1, dt],[0,.96]])  # linear operator

# initial conditions 
psi0 = np.array([0.0, 1.0])        # position, velocity, acceleration 

# Model error   # this is only used to run the 'true' model
## covariance 
C_qq = [[.01,0],     
        [0,.01]]   
## mean 
q_mn = [0,0]

# Measurement errors 
## covariance 
C_ee = [[.01,0],    # 
        [0,1.]] 

## mean
e_mn = [0,0]
e = np.random.multivariate_normal(e_mn, C_ee)
obs_freq = 4  # every 10s

#Initial Forecast Error Covariance... (i think we need this)
C_aa = [[.001,0],     
        [0,.001]]

# Identity matrix 
I = np.eye(2,2)

# 'True' solution 
psi_T = np.empty((tlen,2)) # three rows by tlen times 
psi_T[0,:] = psi0         # set the initial condition
for i in range(1,tlen):
    psi_T[i,:] = G.dot(psi_T[i-1,:])

# Storage vectors
psi_D = np.empty((tlen,2))   # the measurement vector 
psi_F = np.empty((tlen, 2))  # the 'forecast' vector
psi_A = np.empty((tlen, 2))  # the 'analysis' vector 

# set the initial analysis to the initial conition
psi_a = psi0

# Move forward thru time 
for k in range(tlen):
    # random multivariate normal 
    q = np.random.multivariate_normal(q_mn, C_qq)
    
    # run the forward model
    psi_f = G.dot(psi_a) + q 
    
    # Compute the forecast error covariance 
    C_ff = G * C_aa * G.T + C_qq 

    # compute the Kalman gain 
    K = C_ff*M.T*np.linalg.inv((M*C_ff*M.T + C_ee))
    
    # Compute the analysis covariance matrix 
    C_aa = (I - K*M)*C_ff    # we do this even when we don't have a datapoint!

    # set the 'analysis' state to the forecast-- this gets overwritten if data comes in
    psi_a = psi_f 
    d = np.nan

    # perform the filtering --- observations are made 
    if k%obs_freq == 0:   # if the timestep is perfectly divisible by 10
        
        # perfom an observation
        e = np.random.multivariate_normal(e_mn, C_ee)
        d = M.dot(psi_T[k,:]) + e # + the meaasurement noise??
        
        # compute the analysis 
        psi_a = psi_f + K.dot(d - M.dot(psi_f))
       
    # write out data  
    psi_A[k,:] = psi_a
    psi_F[k,:] = psi_f
    psi_D[k,:] = d 

time = np.array([i*dt for i in range(tlen)])
plt.clf()
plt.plot(time, psi_F[:,0], label='forecast')
plt.plot(time, psi_A[:,0], label='analysis')
plt.plot(time, psi_T[:,0], label='Truth')
plt.scatter(time, psi_D[:,0], label='measurement')
plt.xlabel('x position')
plt.ylabel('time (s)')
plt.ylim(0,3.0)
plt.legend()
plt.show() 
