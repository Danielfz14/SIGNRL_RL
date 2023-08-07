# -*- coding: utf-8 -*-
"""SIGNRL_algorithm
"""
##RED
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
import pickle
import concurrent.futures
import time, math, random
from typing import Tuple
from IPython import display

# import gym 
import gym

# Feedforward Network
def feedforward(X, p):
    
    # Roll-back the weights and biases
    weights = model.get_weights()
    b=0
    a=0
    for i in range(len(layers)-1):
    	a=b+int(layers[i]*layers[i+1]) 
    	weights[2*i] = p[b:a].reshape((layers[i],layers[i+1]))
    	b=a+int(layers[i+1])
    	weights[(2*i)+1] = p[a:b].reshape((layers[i+1],))
    model.set_weights(weights)
    logits = model(X)[0,:] 
    return logits          

# Class Agent

class agent:

    # init    
        def __init__(self):
            self.env = gym.make('CartPole-v1')
            self.env.seed(5)

        def episode (self, params, particle):
            alfa = 100/0.21
            beta = 5/2.4
            t_max = 200
            X = np.zeros([1,4])
            obs = self.env.reset()
            X[0,0] = obs[0]
            X[0,1] = obs[1]
            X[0,2] = obs[2]
            X[0,3] = obs[3]
            done = False
            T = 0
            R = []
            discounted_sum = 0
            p = 0
            while ((not done) and (T < t_max)):
                cond_init_ang = np.abs(X[0,2])
                cond_init_pos = np.abs(X[0,0])
                sigmo = feedforward(X,params)
                action = 20 * float(sigmo)
                a = [action, p]
                obs, reward, done, _ = self.env.step(a)
                X[0,0] = obs[0]
                X[0,1] = obs[1]
                X[0,2] = obs[2]
                X[0,3] = obs[3]
                cond_fin_ang = np.abs(X[0,2])
                cond_fin_pos = np.abs(X[0,0])
    			
    			## Reward
                re = reward + (alfa*(cond_init_ang - cond_fin_ang)) + (beta*(cond_init_pos - cond_fin_pos))
                ## Cumulative Reward
                discounted_sum += re
                T += 1 
            
            loss = -discounted_sum
    		
            return loss, particle

# Iteration Function
def f(x):
    
    j = [None for i in range(NP)]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=28) as executor:
    	future_agents = {executor.submit(agentes[i].episode,x[i],i) for i in range(NP)}
    	for future in concurrent.futures.as_completed(future_agents):
    		loss, part = future.result()
    		j[part] = loss
    
    return np.array(j)



if __name__ == "__main__":        
	
    #########################################################################

    #### Training
    layers = [4,20,1]
    NP = 30
    NI = 500
    N_rep = 50
	
	# Loop
    for k in range(N_rep):
		
        print('Execution: '+ str(k))
        agentes = [agent() for x in range(NP)]
	
        #FNN 
        model = Sequential()
        model.add(Dense(layers[1], activation='tanh', input_shape=(layers[0],)))
        model.add(Dense(layers[2], activation='tanh'))
	
		
        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # Call instance of PSO
        dimensions = 0
        for l in range(len(layers)-1):
            dimensions += (layers[l] * layers[l+1]) + layers[l+1] 
        optimizer = ps.single.GlobalBestPSO(n_particles=NP, dimensions=dimensions, options=options)
        # Perform optimization
        cost, pos = optimizer.optimize(f, iters=NI)
        np.savetxt('Train_Results/W_Cartpole_'+str(k)+'.txt',pos)
        with open ('Train_Results/recom_Carpole_'+str(k), "wb") as fp:
            pickle.dump(optimizer.cost_history,fp)

		
        #########################################################################
		
        ## Test

        # '0' ----> Short Episode without Disturbances
        # '1' ----> Long Episode without Disturbances
        # '2' ----> Long Episode with Disturbances
        
        for typ_test in range(3): 

            if (typ_test==0):
                L = 200
                file = 'ShortEpisode_without_Disturbance'
            if (typ_test==1):
                L = 2000
                file = 'LongEpisode_without_Disturbance'
            if (typ_test==2):
                L = 2000
                file = 'LongEpisode_with_Disturbance'

            resultados = np.zeros([50])
            Xtest = np.zeros([1,4])
            position = [10*np.ones([L+1]) for x in range(50)]
            acciones = [np.zeros([L]) for x in range(50)]
            p_angle = [5*np.ones([L+1]) for x in range(50)]
            
            env = gym.make('CartPole-v1')
            
            for y in range(50):
                obs = env.reset()
                position[y][0] = obs[0]
                p_angle[y][0] = obs[2]
                acum = 0
                done = False
                cont = 0
                p = 0
                pert = np.zeros([L+1])    
                while((not done) and (cont<L)):
                    if (typ_test==2):
                        if (cont==200 or cont==225 or cont==250): 
                            p = 15 #(cont-200)*0.3  
                        else:
                            if (cont>700 and cont<750):
                                p = -8
                            else:
                                if (cont==1300 or cont==1325 or cont==1350): ##(cont > 1300 and cont < 1350):
                                    p = -15 #(cont-1300)*-0.3
                                else:
                                    if (cont > 1700 and cont < 1750):
                                        p = 8
                                    else:
                                        p = 0
                    
                    Xtest[0,0] = obs[0]
                    Xtest[0,1] = obs[1]
                    Xtest[0,2] = obs[2]
                    Xtest[0,3] = obs[3]
                    sigmo = feedforward(Xtest, pos)
                    action = 20 * float(sigmo)
                    acciones[y][cont] = action
                    a = [action, p] 
                    pert[cont] = p
                    obs, reward, done, info = env.step(a) 
                    acum += reward
                    cont += 1
                    position[y][cont] = obs[0]
                    p_angle[y][cont] = obs[2]
                resultados[y] = acum
            
            env.close()
            
            with open(file + '/posicion_'+str(k), "wb") as fp:
                pickle.dump(position, fp)
            
            with open(file + '/angulo_'+str(k), "wb") as fp:
                pickle.dump(p_angle, fp)
            
            with open(file + '/acciones_'+str(k), "wb") as fp:
                pickle.dump(acciones, fp)
            
            if (typ_test==2):
                with open(file + '/perturbacion_'+str(k), "wb") as fp:
                    pickle.dump(pert, fp)


        keras.backend.clear_session()
        del agentes
		












