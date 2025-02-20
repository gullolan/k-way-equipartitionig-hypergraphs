#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from gurobipy import*
from gurobipy import GRB
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

#CONSTANTES

TIEMPO_MAX=3600
NOREL_TIME=0
SEMILLA_GRB=1
IMPRIMIR_CONSOLA=0

# In[ ]:


# Función para generación aleatoria de hipergrafos
def rand_hgraf(n,m,d):
    E = np.random.randint(2,d+1,size=(m))
    M = np.arange(m)
    N = np.arange(n)
    A = np.zeros((n,m))
    for j in M:
        t = 0
        while t < E[j]:
            s = int(np.random.choice(N,size=1))
            A[s,j] = 1
            t = np.sum(A[:,j])
    return A


# In[ ]:


#Tamaño promedio de las hiperaristas
def prom(A):
    m = A.shape[1]
    P = []
    for y in range(m):
        P.append(sum(A[:,y]))
    p = np.average(P)
    return p

#Tamaño máximo de las hiperaristas
def tamax(A):
    S = np.sum(A,axis=0)
    m = int(np.max(S))
    return m
#Vértices cubiertos por conjunto de hiperaristas h
def vert_cover(A,h):
    n = A.shape[0]
    S= np.zeros(n)
    for j in h:
        S += A[:,j]
    vc = np.where(S!=0)[0]
    return len(vc)
#Vértices en la interesección de las hiperaristas
def inter(A,e1,e2):
    I = A[:,e1]+A[:,e2]
    i = np.where(I>1)[0]
    return i


# In[ ]:


# Funciones para desigualdades válidas
#Hiperaristas de vértices tipo hoja
def hoja(A):
    n = A.shape[0]
    S = np.sum(A,axis=1)
    h = []
    for i in range(n):
        if S[i]==1:
            j = np.where(A[i,:]==1)[0][0]
            h.append(j)
    return h
#Hiperaristas contenidas
def cont(A):
    m = A.shape[1]
    tmax = tamax(A)
    h = {}
    for t in range(2,tmax+1):
        E = []
        for j in range(m):
            if sum(A[:,j])==t:
                E.append(j)
        h.update({t: E})
    C = {}
    for c_out in range(tmax,2,-1):
        for eo in h[c_out]:
            L = []
            for c_in in range(c_out-1,1,-1):
                for ei in h[c_in]:
                    d = A[:,ei]+A[:,eo]
                    s = len(np.where(d==2)[0])
                    if s == c_in:
                        L.append(ei)
            if len(L)>0:
                C.update({eo:L})
    return C
#Conjuntos minimales de 3 elementos
def minimal3(A,nk):
    m = A.shape[1]
    M = [x for x in range(m)]
    S = []
    for x in M:
        s = []
        i = M.index(x)
        for y in M[i+1:]:
            i = M.index(y)
            for z in M[i+1:]:
                if vert_cover(A,[x,y])<=nk and vert_cover(A,[x,z])<=nk and vert_cover(A,[y,z])<=nk and vert_cover(A,[x,y,z])>nk:
                    S.append([x,y,z])
                else:
                    break
    return S


# ## Formulación $\mathcal{F}_1$

# In[ ]:


def F1(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1

    modelo.Params.NoRelHeurTime=NOREL_TIME

    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
    
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,nods,g


# ## Formulación $\mathcal{F}_1$ con planos cortantes

# In[ ]:


#Restricción tipo hoja
def F1_pc1(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Hiperaristas tipo hoja
    for h in hoja(A):
        modelo.addConstr(quicksum(y[h,k] for k in C) == 1)
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,nods,g


# In[ ]:


#Eliminación de simetrías
def F1_pc2(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Eliminación de simetrías
    for c in C:
        modelo.addConstr(quicksum(x[c,l] for l in range(c+1,K))==0)
        modelo._cbCuts+=1
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
    
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#Hiperaristas contenidas
def F1_pc3(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Hiperaristas contenidas
    y_node = {(j,k): 0 for j in M for k in C}
    CONT = cont(A)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for k in C:
                    for j in M:
                        y_node[j,k] = modelo.cbGetNodeRel(y[j,k])
                for p in CONT.keys():
                    for q in CONT[p]:
                        ds = quicksum(y_node[p,k]+y_node[q,k] for k in C)
                        if  ds.getValue()>1:
                            modelo.cbCut(quicksum(y[p,k]+y[q,k] for k in C) <= 1)
                            modelo._cbCuts+=1
                        else:
                            pass
    modelo.update()
    modelo.optimize(mycallback)
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods=modelo.NodeCount
        g = 100*modelo.MIPGap
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#Restricción tipo knapsack
def F1_pc4(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Desigualdades tipo knapsack
    y_node = {(j,k): 0 for j in M for k in C}
    Min3 = minimal3(A,nk)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for k in C:
                    for j in M:
                        y_node[j,k] = modelo.cbGetNodeRel(y[j,k])
                for s in Min3:
                    for k in C:
                        ds = quicksum(y_node[j,k] for j in s)
                        if ds.getValue()>len(s)-1:
                            modelo.cbCut(quicksum(y[j,k] for j in s) <= len(s)-1)
                            modelo._cbCuts+=1
                    else:
                        pass
    modelo.update()
    modelo.optimize(mycallback)
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#Combinación ES+CO
def F1_pc_23(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Eliminación de simetrías
    for c in C:
        modelo.addConstr(quicksum(x[c,l] for l in range(c+1,K))==0)
        modelo._cbCuts+=1
    #Hiperaristas contenidas
    y_node = {(j,k): 0 for j in M for k in C}
    CONT = cont(A)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for k in C:
                    for j in M:
                        y_node[j,k] = modelo.cbGetNodeRel(y[j,k])
                for p in CONT.keys():
                    for q in CONT[p]:
                        ds = quicksum(y_node[p,k]+y_node[q,k] for k in C)
                        if  ds.getValue()>1:
                            modelo.cbCut(quicksum(y[p,k]+y[q,k] for k in C) <= 1)
                            modelo._cbCuts+=1
                        else:
                            pass
    modelo.update()
    modelo.optimize(mycallback)
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#Combinación ES+KS
def F1_pc_24(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Eliminación de simetrías
    for c in C:
        modelo.addConstr(quicksum(x[c,l] for l in range(c+1,K))==0)
        modelo._cbCuts+=1
    #Desigualdades tipo knapsack
    y_node = {(j,k): 0 for j in M for k in C}
    Min3 = minimal3(A,nk)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for k in C:
                    for j in M:
                        y_node[j,k] = modelo.cbGetNodeRel(y[j,k])
                for s in Min3:
                    for k in C:
                        ds = quicksum(y_node[j,k] for j in s)
                        if ds.getValue()>len(s)-1:
                            modelo.cbCut(quicksum(y[j,k] for j in s) <= len(s)-1)
                            modelo._cbCuts+=1
                    else:
                        pass
    modelo.update()
    modelo.optimize(mycallback)
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#Combinación CO+KS
def F1_pc_34(A,W,K):
    #Parámetros
    #A: matriz de incidencia del hipergrafo
    #W: vector de pesos de las hiperaristas
    #K: número de particionamiento
    n = A.shape[0]   #Número de nodos
    m = A.shape[1]   #Número de hiperaristas
    nk = int(n/K)     #Tamaño de cada partición
    #Listas de conjuntos para las variables
    N = np.arange(n)
    M = np.arange(m)
    C = np.arange(K)
    #Creación del modelo
    modelo=Model()
    modelo.Params.LogToConsole = IMPRIMIR_CONSOLA
    modelo.Params.timeLimit = TIEMPO_MAX
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.NodefileStart=0.5
    modelo.Params.PreCrush = 1
    modelo.Params.NoRelHeurTime=NOREL_TIME
    modelo.Params.Seed=SEMILLA_GRB

    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N) #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C) #(2.3)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N) #(2.4)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M) #(2.5)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C) #(2.6)
    #Hiperaristas contenidas y Desigualdades tipo knapsack
    y_node = {(j,k): 0 for j in M for k in C}
    CONT = cont(A)
    Min3 = minimal3(A,nk)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for k in C:
                    for j in M:
                        y_node[j,k] = modelo.cbGetNodeRel(y[j,k])
                for p in CONT.keys():
                    for q in CONT[p]:
                        ds = quicksum(y_node[p,k]+y_node[q,k] for k in C)
                        if  ds.getValue()>1:
                            modelo.cbCut(quicksum(y[p,k]+y[q,k] for k in C) <= 1)
                            modelo._cbCuts += 1
                        else:
                            pass
                for s in Min3:
                    for k in C:
                        ds = quicksum(y_node[j,k] for j in s)
                        if ds.getValue()>len(s)-1:
                            modelo.cbCut(quicksum(y[j,k] for j in s) <= len(s)-1)
                            modelo._cbCuts += 1
                    else:
                        pass
    modelo.update()
    modelo.optimize(mycallback)
    #Recuperación de variables
    V = []
    E = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return V,E,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for k in C:
            X = []
            for i in N:
                if x[i,k].x != 0:
                    X.append(i)
            V.append(X)
            Y = []
            for j in M:
                if y[j,k].x != 0:
                    Y.append(j)
            E.append(Y)
    return V,E,time_r,sol,[nods,modelo._cbCuts],g




def leer_instancia(nombre_archivo, delimitador=" "):
    """
    Lee las dimensiones, un vector de pesos y la matriz de incidencia.
    
    Entrada:
        nombre_archivo (str): El nombre del archivo
        delimitador (str): El separador entre elementos (por defecto, espacio).

    Salida:
        tuple: Una tupla con las dimensiones (filas, columnas), el vector de pesos, y la matriz.
    """
    try:
        with open(nombre_archivo, "r") as archivo:
            # Leer las dimensiones de la matriz
            dimensiones = archivo.readline().strip().split(delimitador)
            filas, columnas, k = map(int, dimensiones)
            
            # Leer el vector de pesos
            pesos = np.array(list(map(float, archivo.readline().strip().split(delimitador))))

            # Leer la matriz
            matriz = np.array([list(map(float, archivo.readline().strip().split(delimitador))) for _ in range(filas)])

            return filas, columnas, k, pesos, matriz
    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no existe.")
    except ValueError as e:
        print(f"Error al procesar los datos: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    return None

########################
######FUNCIÓN PRINCIPAL
########################


print("\n **********************************************************")
print("\t \t COMPUTATIONAL TESTS") 
print("***********************************************************")
print("\t k-way Hypergraphs Equipartitiong: F1 Formulation \n\n")

Instancia = np.array(["45_500_2_3.txt", 
                      "45_500_2_5.txt",
                      "45_500_5_3.txt",
                      "45_500_5_5.txt",
                      "45_500_10_3.txt",
                      "45_500_10_5.txt",
                      "90_1000_2_3.txt",
                      "90_1000_2_5.txt",
                      "90_1000_5_3.txt",
                      "90_1000_5_5.txt",
                      "90_1000_10_3.txt",
                      "90_1000_10_5.txt",
                      "150_1500_2_3.txt",
                      "150_1500_2_5.txt",
                      "150_1500_5_3.txt",
                      "150_1500_5_5.txt",
                      "150_1500_10_3.txt",
                      "150_1500_10_5.txt",
                      "180_2000_2_3.txt",
                      "180_2000_2_5.txt",
                      "180_2000_5_3.txt",
                      "180_2000_5_5.txt",
                      "180_2000_10_3.txt",
                      "180_2000_10_5.txt"
                    ])

R = []


for p in Instancia:
    Nombre="Instances/"+p

    
    n,m,K,W,A=leer_instancia(Nombre," ")
    sol =None
    t=tamax(A)
    pr = prom(A)

    if t==2 and m/n <3:
        V,E,tr,sol,nod,g = F1(A,W,K)
        V1,E1,tr1,sol1,nod1,g1 = F1_pc1(A,W,K)
        V2,E2,tr2,sol2,nod2,g2 = F1_pc2(A,W,K)
        V3,E3,tr3,sol3,nod3,g3 = None,None,None,None,None,None
        V4,E4,tr4,sol4,nod4,g4 = F1_pc4(A,W,K)
        V23,E23,tr23,sol23,nod23,g23 = F1_pc_23(A,W,K)
        V24,E24,tr24,sol24,nod24,g24 = F1_pc_24(A,W,K)
        V34,E34,tr34,sol34,nod34,g34 = F1_pc_34(A,W,K)
    elif t==2 and m/n >3:
        V,E,tr,sol,nod,g = F1(A,W,K)
        V1,E1,tr1,sol1,nod1,g1 = None,None,None,None,None,None
        V2,E2,tr2,sol2,nod2,g2 = F1_pc2(A,W,K)
        V3,E3,tr3,sol3,nod3,g3 = None,None,None,None,None,None
        V4,E4,tr4,sol4,nod4,g4 = F1_pc4(A,W,K)
        V23,E23,tr23,sol23,nod23,g23 = F1_pc_23(A,W,K)
        V24,E24,tr24,sol24,nod24,g24 = F1_pc_24(A,W,K)
        V34,E34,tr34,sol34,nod34,g34 = F1_pc_34(A,W,K)
    else:
        V,E,tr,sol,nod,g = F1(A,W,K)
        V1,E1,tr1,sol1,nod1,g1 = None,None,None,None,None,None
        V2,E2,tr2,sol2,nod2,g2 = F1_pc2(A,W,K)
        V3,E3,tr3,sol3,nod3,g3 = F1_pc3(A,W,K)
        V4,E4,tr4,sol4,nod4,g4 = F1_pc4(A,W,K)
        V23,E23,tr23,sol23,nod23,g23 = F1_pc_23(A,W,K)
        V24,E24,tr24,sol24,nod24,g24 = F1_pc_24(A,W,K)
        V34,E34,tr34,sol34,nod34,g34 = F1_pc_34(A,W,K)

    print("===> Computing ", Nombre)
    
    Tr=[tr,tr1,tr2,tr3,tr4,tr23,tr24,tr34]
    S=[sol,sol1,sol2,sol3,sol4,sol23,sol24,sol34]
    Nd=[nod,nod1,nod2,nod3,nod4,nod23,nod24,nod34]
    G=[g,g1,g2,g3,g4,g23,g24,g34]
    R.append([p,pr,t,Tr,S,Nd,G])
    
    Df = pd.DataFrame(columns=['Instance','Avg','MaxSize','Cuts','ObjValue','Gap', 'NodesBB','Time'],
                 index=np.arange(len(R)*8))
    # pc = ['NA','TH','ES','CO','KS','ES+CO','ES+KS','CO+KS']
    pc = ['NA','Th-3','Th-4','Th-5','Th-6','Th-4+Th-5','Th-4+Th-6','Th-5+Th-6']


    for x in range(len(R)):
        for y in range(8):
            Df.at[8*x+y,'Instance']=R[x][0]
            Df.at[8*x+y,'Avg']=R[x][1]
            Df.at[8*x+y,'MaxSize']=R[x][2]
            Df.at[8*x+y,'Cuts']=pc[y]
            Df.at[8*x+y,'ObjValue']=R[x][4][y]
            Df.at[8*x+y,'Gap']=R[x][6][y]
            Df.at[8*x+y,'NodesBB']=R[x][5][y]
            Df.at[8*x+y,'Time']=R[x][3][y]

    Df.to_excel('Tests_F1_cuts.xlsx')


