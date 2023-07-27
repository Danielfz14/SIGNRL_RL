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

        def episode (self, params, particle):
            alfa = 1000
            beta = 10
            t_max = 200
            gamma = 1
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
            while ((not done) and (T < t_max)):
                cond_init_ang = np.abs(X[0,2])
                cond_init_pos = np.abs(X[0,0])
                sigmo = feedforward(X,params)
                #action=np.argmax(sigmo)
                action = 20 * float(sigmo)
                obs, reward, done, _ = self.env.step(action)
                X[0,0]=obs[0]
                X[0,1]=obs[1]
                X[0,2]=obs[2]
                X[0,3]=obs[3]
                cond_fin_ang = np.abs(X[0,2])
                cond_fin_pos = np.abs(X[0,0])
    			
    			## Recompensa resultados 5
                re = reward + (alfa*(cond_init_ang - cond_fin_ang)) + (beta*(cond_init_pos - cond_fin_pos))
    			
    			## Recompensas resultados 4
    			#re=0
    			#if (T<20 and np.abs(obs[2])<0.2):
    			#	re = ((cond_init*180)/(math.pi))
    			#if (T > 20 and np.abs(obs[2])<0.2):
    			#	re = 0.1
    			#if (T > 20 and np.abs(obs[2])<0.05):
    			#	re = 0.25
    			#if (T > 150 and np.abs(obs[2])<0.05 and np.abs(obs[0])<1):
    			#	re = 2 - np.abs(obs[0]) 
    				
    			#if (T<20 and np.abs(obs[2])<0.2):
    			#	re = ((cond_init*180)/(12*math.pi))
    			#if (T > 20 and np.abs(obs[2])<0.2):
    			#	re = (1-((np.abs(obs[2])*180)/(12*math.pi)))
    			#if (T > 60 and np.abs(obs[2])<0.05 and np.abs(obs[0])<1):
    			#	re = 4 - np.abs(obs[0])
    			
                discounted_sum += re
    			#(1-((np.abs(obs[2])*180)/(12*math.pi)))
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
    NI=200
    N_rep=50
	
    
    start_time = t()
	
	# Loop
    for k in range(N_rep):
		
        print('Ejecucion Nro: '+ str(k))
	
		#Red 
        model = Sequential()
        model.add(Dense(layers[1], activation='tanh', input_shape=(layers[0],)))
        model.add(Dense(layers[2], activation='tanh'))
	
		
		# Initialize agent
        pos = np.loadtxt('Resultados_New_4/W_Cartpole_'+str(k)+'.txt')
		## Prueba
        resultados = np.zeros([50])
        Xtest=np.zeros([1,4])
        position = [2.5*np.ones([2001]) for x in range(50)]
        acciones = [np.zeros([2000]) for x in range(50)]
        p_angle = [0.5*np.ones([2001]) for x in range(50)]
		
        env = gym.make('CartPole-v1')
		
        for y in range(50):
            obs = env.reset()
            position[y][0]=obs[0]
            p_angle[y][0]=obs[2]
            acum=0
            done=False
            cont=0
            p = 0
            p2 = -8
            pert = np.zeros([2001])
            while(not done):
                if (cont>200 and cont<250): ###:
            		#obs, reward, done, info = env.step(p)
            		#cont += 1
            		#position[y][cont]=obs[0]
            		#p_angle[y][cont]=obs[2]
            		#pert [cont] = p
            		#p += 1
                    p=(cont-200)*0.3  ##
                else:
                    if (cont>700 and cont<750):
                        p = -8
                    else:
                        if (cont > 1300 and cont < 1350): ##(cont > 1300 and cont < 1350):
                            p = (cont-1300)*-0.3 #
                        else:
                            if (cont > 1700 and cont < 1750):
                                p = 8
                            else:
                                p = 0
            	
            
            	
                Xtest[0,0]=obs[0]
                Xtest[0,1]=obs[1]
                Xtest[0,2]=obs[2]
                Xtest[0,3]=obs[3]
                sigmo = feedforward(Xtest, pos)
                action = 20 * float(sigmo)
                acciones[y][cont]=action
                a = [action, p] 
                pert[cont]=p
                obs, reward, done, info = env.step(a) 
                acum+=reward
                cont+=1
                position[y][cont]=obs[0]
                p_angle[y][cont]=obs[2]
            resultados[y]=acum
        
        env.close()
        
        with open('Resultados_New_conpert2/posicion_'+str(k), "wb") as fp:
            pickle.dump(position, fp)
		
        with open('Resultados_New_conpert2/angulo_'+str(k), "wb") as fp:
            pickle.dump(p_angle, fp)
		
        with open('Resultados_New_conpert2/acciones_'+str(k), "wb") as fp:
            pickle.dump(acciones, fp)
        
        with open('Resultados_New_conpert2/perturbacion_'+str(k), "wb") as fp:
            pickle.dump(pert, fp)
			
        #fig, ax = plt.subplots(4)
		
        #for i in range(50):
        #	ax[0].plot(position[i])
        #	ax[0].axis([0,2000,-2.5,2.5])
        #	ax[0].set_xlabel('Time-step')
        #	ax[0].set_ylabel('Position')
        #	ax[1].plot(p_angle[i])
        #	ax[1].axis([0,2000,-0.5,0.5])
        #	ax[1].set_xlabel('Time-step')
        #	ax[1].set_ylabel('Angle')
        #	ax[2].plot(acciones[i])
        #	ax[2].axis([0,2000,-25,25])
        #	ax[2].set_xlabel('Time-step')
        #	ax[2].set_ylabel('Force (N)')
        #	if (i == 0):
        #		ax[3].plot(pert)
        #		#ax[3].axis([0,500,-25,25])
        #		ax[3].set_xlabel('Time-step')
        #		ax[3].set_ylabel('Force (N)')
		
        #plt.show()
		
        keras.backend.clear_session()
        
		

	#elapsed_time = t() - start_time
	#print("Tiempo de entrenamiento: %0.10f segundos." % elapsed_time) 
	    











