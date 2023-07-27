# -*- coding: utf-8 -*-
"""Contrl_RL_Keras
"""
##RED
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
import random
import matplotlib.pyplot as plt
import pickle
import asyncio
import concurrent.futures
from time import time as t
import time, math, random
from typing import Tuple
from IPython import display

# import gym 
import gym

# Feedforward Network
def feedforward(X, p):
    
    # Roll-back the weights and biases
    pesos = model.get_weights()
    b=0
    a=0
    for i in range(len(layers)-1):
    	a=b+int(layers[i]*layers[i+1]) 
    	pesos[2*i] = p[b:a].reshape((layers[i],layers[i+1]))
    	b=a+int(layers[i+1])
    	pesos[(2*i)+1] = p[a:b].reshape((layers[i+1],))
    model.set_weights(pesos)
    logits = model(X)[0,:] # Pre-activation in Layer 2
    return logits          # Logits for Layer 2

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
            obs=self.env.reset()
            X[0,0]=obs[0]
            X[0,1]=obs[1]
            X[0,2]=obs[2]
            X[0,3]=obs[3]
            done=False
            T=0
            R=[]
            discounted_sum=0
            p=0
            while ((not done) and (T < t_max)):
                cond_init_ang = np.abs(X[0,2])
                cond_init_pos = np.abs(X[0,0])
                sigmo = feedforward(X,params)
                #action=np.argmax(sigmo)
                action = 20 * float(sigmo)
                a = [action, p]
                obs, reward, done, _ = self.env.step(a)
                X[0,0]=obs[0]
                X[0,1]=obs[1]
                X[0,2]=obs[2]
                X[0,3]=obs[3]
                cond_fin_ang = np.abs(X[0,2])
                cond_fin_pos = np.abs(X[0,0])
    			
    			## Recompensa
                re = reward + (alfa*(cond_init_ang - cond_fin_ang)) + (beta*(cond_init_pos - cond_fin_pos))
    			
    			discounted_sum += re
    			T += 1 
            
            loss=-discounted_sum
    		
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
	#### Training

    layers=[4,20,1]
    NP=30
    NI=500
    N_rep=50
	
    
    start_time = t()
	
	# Loop
    for k in range(N_rep):
		
        print('Ejecucion Nro: '+ str(k))
        agentes = [agent() for x in range(NP)]
	
        #Red 
        model = Sequential()
        model.add(Dense(layers[1], activation='tanh', input_shape=(layers[0],)))
        model.add(Dense(layers[2], activation='tanh'))
	
		
        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # Call instance of PSO
        dimensions=0
        for l in range(len(layers)-1):
            dimensions += (layers[l] * layers[l+1]) + layers[l+1] 
        optimizer = ps.single.GlobalBestPSO(n_particles=NP, dimensions=dimensions, options=options)
        # Perform optimization
        cost, pos = optimizer.optimize(f, iters=NI)
        np.savetxt('Resultados_New_5/W_Cartpole_'+str(k)+'.txt',pos)
        #np.savetxt('Resultados_New/recom_Cartpole_'+str(k)+'.txt',optimizer.cost_history)
        with open ('Resultados_New_5/recom_Carpole_'+str(k), "wb") as fp:
            pickle.dump(optimizer.cost_history,fp)

		
		## Prueba
        resultados = np.zeros([50])
        Xtest=np.zeros([1,4])
        position = [2.5*np.ones([501]) for x in range(50)]
        acciones = [np.zeros([500]) for x in range(50)]
        p_angle = [0.5*np.ones([501]) for x in range(50)]
		
        env = gym.make('CartPole-v1')
		
        for y in range(50):
            obs = env.reset()
            position[y][0]=obs[0]
            p_angle[y][0]=obs[2]
            acum=0
            done=False
            cont=0
            p=0
            while(not done):
                Xtest[0,0]=obs[0]
                Xtest[0,1]=obs[1]
                Xtest[0,2]=obs[2]
                Xtest[0,3]=obs[3]
                sigmo = feedforward(Xtest, pos)
                #action = np.argmax(sigmo)
                action = 20 * float(sigmo)
                a = [action, p]
                acciones[y][cont]=action
                obs, reward, done, info = env.step(a) 
                acum+=reward
                cont+=1
                position[y][cont]=obs[0]
                p_angle[y][cont]=obs[2]
            resultados[y]=acum
		
        env.close()
		
        with open('Resultados_New_5/posicion_'+str(k), "wb") as fp:
            pickle.dump(position, fp)
		
        with open('Resultados_New_5/angulo_'+str(k), "wb") as fp:
            pickle.dump(p_angle, fp)
		
        with open('Resultados_New_5/acciones_'+str(k), "wb") as fp:
            pickle.dump(acciones, fp)
		
		
        keras.backend.clear_session()
        del agentes
		












