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


#Hiperaristas que forman ciclos
def find_cycles(graph):
    def dfs(node,parent,visited,current_cycle):
        visited[node] = True
        current_cycle.append(node)
        for neighbor in graph[node]:
            if neighbor != parent:
                if visited[neighbor]:
                    if neighbor in current_cycle:
                        cycles.append(current_cycle[current_cycle.index(neighbor):])
                else:
                    dfs(neighbor, node, visited, list(current_cycle))
    visited = {node: False for node in graph}
    cycles = []
    for node in graph:
        if not visited[node]:
            dfs(node, None, visited, [])
    return cycles

def ciclos(A,P):
    n_int = set([inter(A,x,y)[0] for x in P for y in P if len(inter(A,x,y))>0 and x!=y])
    adjv = {i: [] for i in n_int}
    for x in P:
        for y in P:
            if x!=y:
                if len(inter(A,x,y))>0:
                    adjv[inter(A,x,y)[0]].append('e{}'.format(x))
    for x in adjv:
        adjv[x]=set(adjv[x])
    part = ['e{}'.format(x) for x in P]
    adjh = {x:[y for y in adjv.keys() if x in adjv[y]] for x in part}
    #grafo formado por las hiperaristas de P y los nodos en sus intersecciones
    graph = {}
    graph.update(adjh)
    graph.update(adjv)
    cy = find_cycles(graph)
    return [[x for x in cy[y] if type(x)==str] for y in range(len(cy))]


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
#Hiperárboles minimales
def minhtree(A,nk,count):
    LIM = 100
    h_inc = {y:[x for x in range(m) if len(inter(A,x,y))==1] for y in range(m)}
    CM = []
    i=0
    j=0
    while i<count and j<LIM:
        M = [y for y in range(m)]
        random.shuffle(M)
        hi = M.pop(0)
        P = []
        while vert_cover(A,P)<= nk:
            #print('hi={}'.format(hi))
            P.append(hi)
            #print('P={}'.format(P))
            for pi in h_inc[hi]:
                if pi not in P:
                    #print('pi={}'.format(pi))
                    nodint=[inter(A,pi,x)[0] for x in P if len(inter(A,pi,x))!=0]
                    #print(nodint)
                    if len(set(nodint))==1:
                        hi = pi
                        #print('hipi={}'.format(hi))
                        break
            if hi in P:
                hi = M.pop(0)
                P = []
        INC = {x:set([inter(A,y,x)[0] for y in P if y!=x and len(inter(A,y,x))==1]) for x in P}
        #print('INC={}'.format(INC))
        j+=1
        if vert_cover(A,P[1:])<=nk and len([1 for y in CM if set(P)==set(y[0])])==0:
            CM.append([P,[x for x in INC.keys() if len(INC[x])==1]])
            #print('CM={}'.format(CM))
            i+=1
    return CM
#Intersección mayor a 1
def int1(A):
    m = A.shape[1]
    I = []
    for p in range(m):
        for q in range(p+1,m):
            if len(inter(A,p,q))>1:
                I.append([p,q])
    return I
#Número de hiperaristas mínimo
def ALFA(A,nk):
    T=list(set(np.sum(A,axis=0).astype(int)))
    T.sort(reverse=True)
    t = 100
    i = 0
    r = []
    C = {}
    while t>2:
        t = T[i]
        d,e = math.modf((nk-1-sum(r))/(t-1))
        C.update({t:e})
        if e!=0 and nk!=t:
            r.append(e*(t-1))
            i += 1
        else:
            i += 1
    C_f = {x:C[x] for x in C.keys() if C[x]>0}
    s = sum([C_f[x] for x in C_f.keys()])
    return C_f,s


# ## Formulación $\mathcal{F}_2$

# In[ ]:


def F2(A,W,K):
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
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600
    modelo.Params.cuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N)
    #(2.3)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C)
    #(2.4)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N)
    #(2.5)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M)
    #(2.6)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C)
    #(2.11)
    for i in range(m):
        for j in range(i+1,m):
            if len(inter(A,i,j))>1:
                modelo.addConstr(quicksum(y[i,k]+y[j,k] for k in C)<= 1)
    #(2.12)
    modelo.addConstrs(quicksum((sum(A[:,j])-1)*y[j,k] for j in M) == nk-1 for k in C)
    modelo.update()
    #Lazy constraints (2.13)
    def cicloelim(modelo,where):
        E_laz = [[] for _ in C]
        if where == GRB.Callback.MIPSOL:
            y_vals = modelo.cbGetSolution(y)
            for k in C:
                for j in M:
                    if y_vals[j,k]>0.5:
                        E_laz[k].append(j)
                CY = ciclos(A,E_laz[k])
                if len(CY)>0:
                    for p in range(len(CY)):
                        S = [int(x[1:]) for x in CY[p]]
                        for l in C:
                            modelo.cbLazy(quicksum(y[j,l] for j in S) <= len(S)-1)
    modelo.Params.lazyConstraints = 1
    modelo.optimize(cicloelim)
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


# ## Formulación $\mathcal{F}_2$ con planos cortantes

# In[ ]:


#Eliminacion de simetrias + Hiperaristas contenidas
def F2_pc1(A,W,K):
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
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N)
    #(2.3)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C)
    #(2.4)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N)
    #(2.5)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M)
    #(2.6)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C)
    #(2.11)
    for i in range(m):
        for j in range(i+1,m):
            if len(inter(A,i,j))>1:
                modelo.addConstr(quicksum(y[i,k]+y[j,k] for k in C)<= 1)
    #(2.12)
    modelo.addConstrs(quicksum((sum(A[:,j])-1)*y[j,k] for j in M) == nk-1 for k in C)
    #Eliminación de simetrías
    for c in C:
        modelo.addConstr(quicksum(x[c,l] for l in range(c+1,K))==0)
        modelo._cbCuts+=1
    y_node = {(j,k): 0 for j in M for k in C}
    CONT = cont(A)
    modelo.update()
    #Lazy constraints (2.13)
    def cicloelim(modelo,where):
        E_laz = [[] for _ in C]
        if where == GRB.Callback.MIPSOL:
            y_vals = modelo.cbGetSolution(y)
            for k in C:
                for j in M:
                    if y_vals[j,k]>0.5:
                        E_laz[k].append(j)
                CY = ciclos(A,E_laz[k])
                if len(CY)>0:
                    for p in range(len(CY)):
                        S = [int(x[1:]) for x in CY[p]]
                        for l in C:
                            modelo.cbLazy(quicksum(y[j,l] for j in S) <= len(S)-1)
    #Hiperaristas contenidas
        elif where == GRB.Callback.MIPNODE:
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
    modelo.Params.lazyConstraints = 1
    modelo.optimize(cicloelim)
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


#Eliminación de simetrías y Restricción tipo knapsack
def F2_pc2(A,W,K):
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
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N)
    #(2.3)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C)
    #(2.4)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N)
    #(2.5)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M)
    #(2.6)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C)
    #(2.11)
    for i in range(m):
        for j in range(i+1,m):
            if len(inter(A,i,j))>1:
                modelo.addConstr(quicksum(y[i,k]+y[j,k] for k in C)<= 1)
    #(2.12)
    modelo.addConstrs(quicksum((sum(A[:,j])-1)*y[j,k] for j in M) == nk-1 for k in C)
    #Eliminación de simetrías
    for c in C:
        modelo.addConstr(quicksum(x[c,l] for l in range(c+1,K))==0)
        modelo._cbCuts+=1
    modelo.update()
    y_node = {(j,k): 0 for j in M for k in C}
    Minh = minhtree(A,nk,10)
    #Lazy constraints (2.13)
    def cicloelim(modelo,where):
        E_laz = [[] for _ in C]
        if where == GRB.Callback.MIPSOL:
            y_vals = modelo.cbGetSolution(y)
            for k in C:
                for j in M:
                    if y_vals[j,k]>0.5:
                        E_laz[k].append(j)
                CY = ciclos(A,E_laz[k])
                if len(CY)>0:
                    for p in range(len(CY)):
                        S = [int(x[1:]) for x in CY[p]]
                        for l in C:
                            modelo.cbLazy(quicksum(y[j,l] for j in S) <= len(S)-1)
        elif where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for k in C:
                    for j in M:
                        y_node[j,k] = modelo.cbGetNodeRel(y[j,k])
                #Desigualdades tipo knapsack
                for s in Minh:
                    for k in C:
                        ds=quicksum(y_node[j,k] for j in s[1])
                        if ds.getValue()>len(s[1])-1:
                            modelo.cbCut(quicksum(y[j,k] for j in s[1]) <= len(s[1])-1 )
                            modelo._cbCuts+=1
    modelo.Params.lazyConstraints = 1
    modelo.optimize(cicloelim)
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


#Eliminación de simetrías y Número mínimo de hiperaristas
def F2_pc3(A,W,K):
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
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    x = modelo.addVars(N,C,vtype=GRB.BINARY, name='x')
    y = modelo.addVars(M,C,vtype=GRB.BINARY, name='y')
    #Función objetivo
    obj = quicksum(W[j]*y[j,k] for j in M for k in C)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    #(2.2)
    modelo.addConstrs(quicksum(x[i,k] for k in C) == 1 for i in N)
    #(2.3)
    modelo.addConstrs(quicksum(x[i,k] for i in N) == nk for k in C)
    #(2.4)
    modelo.addConstrs(x[i,k] <= quicksum(A[i,j]*y[j,k] for j in M) for k in C for i in N)
    #(2.5)
    modelo.addConstrs(quicksum(y[j,k] for k in C) <= 1 for j in M)
    #(2.6)
    modelo.addConstrs(quicksum(A[i,j]*x[i,k] for i in N) >= sum(A[:,j])*y[j,k] for j in M for k in C)
    #(2.11)
    for i in range(m):
        for j in range(i+1,m):
            if len(inter(A,i,j))>1:
                modelo.addConstr(quicksum(y[i,k]+y[j,k] for k in C)<= 1)
    #(2.12)
    modelo.addConstrs(quicksum((sum(A[:,j])-1)*y[j,k] for j in M) == nk-1 for k in C)
    #Eliminación de simetrías
    for c in C:
        modelo.addConstr(quicksum(x[c,l] for l in range(c+1,K))==0)
        modelo._cbCuts+=1
    #Número mínimo de hiperaristas
    modelo.addConstr(quicksum(y[j,k] for j in M for k in C)>=K*ALFA(A,nk)[1])
    modelo._cbCuts+=1
    modelo.update()
    #Lazy constraints (2.13)
    def cicloelim(modelo,where):
        E_laz = [[] for _ in C]
        if where == GRB.Callback.MIPSOL:
            y_vals = modelo.cbGetSolution(y)
            for k in C:
                for j in M:
                    if y_vals[j,k]>0.5:
                        E_laz[k].append(j)
                CY = ciclos(A,E_laz[k])
                if len(CY)>0:
                    for p in range(len(CY)):
                        S = [int(x[1:]) for x in CY[p]]
                        for l in C:
                            modelo.cbLazy(quicksum(y[j,l] for j in S) <= len(S)-1)
    modelo.Params.lazyConstraints = 1
    modelo.optimize(cicloelim)
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


# ## Formulación $\mathcal{F}_3$

# In[ ]:


def F3(A,W,K):
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
    Nv = ['v{}'.format(i) for i in range(n)]
    Me = ['e{}'.format(i) for i in range(m)]
    w = Nv+Me+['S']
    arcs={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arcx={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arci={(j,i): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    A2 = {**arcs,**arci}
    A1={('S',i): nk for i in Nv}
    arcs.update(arci)
    arcs.update(A1)
    d={}
    for x in w:
        if x in Nv:
            d.update({x:1})
        elif x in Me:
            d.update({x:0})
        else:
            d.update({x:-n})
    arcos, cap = multidict(arcs)
    #Creación del modelo    
    modelo=Model()
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600.0
    modelo.Params.cuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    f = modelo.addVars(arcos,vtype=GRB.INTEGER,name='f',lb=0)
    z = modelo.addVars(arcos,vtype=GRB.BINARY, name='z')
    h = modelo.addVars(M,vtype=GRB.BINARY, name='h')
    #Función objetivo
    obj = quicksum(W[j]*h[j] for j in M)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstr(quicksum(z['S',i] for i in Nv) == K) #(2.15)
    modelo.addConstrs(f['S',i] == nk*z['S',i] for i in Nv) #(2.16)
    modelo.addConstrs(z[i,j]<= f[i,j] for (i,j) in A2) #(2.17a)
    modelo.addConstrs(f[i,j] <= (nk-1)*z[i,j] for (i,j) in A2) #(2.17b)
    modelo.addConstrs(f.sum('*',i)-f.sum(i,'*') == d[i] for i in w) #(2.18)
    modelo.addConstrs(z.sum(j,'*') == (sum(A[:,Me.index(j)])-1)*h[Me.index(j)] for j in Me) #(2.19)
    modelo.addConstrs(z.sum('*',j) == h[Me.index(j)] for j in Me) #(2.20)
    modelo.addConstrs(z.sum('*',i) == 1 for i in Nv) #(2.21)
    modelo.addConstrs(z[i,j]+z[j,i]<= 1 for (i,j) in arcx) #(2.22)
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    F = []
    Z = []
    Y = []
    H = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return [],H,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for i in arcos:
            if f[i].x > 0:
                F.append([i,f[i].x])    
        for j in arcos:
            if z[j].x > 0:
                Z.append([j,z[j].x])
                if j[0]=='S':
                    Y.append(j[1])     
        k=len(Y)
        pz=[]
        f=dict(F)
        Fv=[x for x in f if x[0] in Nv]
        Fe=[x for x in f if x[0] in Me]
        for y in Y:
            pz.extend([('S',y)])
            T=[y]
            p=[0]
            pt=[0]
            while len(pt)>0:
                t=[]
                pt = []
                for i in T:
                    if i in Nv:
                        p=[x for x in Fv if x[0]==i]
                        s=[x[1] for x in Fv if x[0]==i]
                        for y in p:
                            Fv.remove(y)
                        pt.extend(p)
                    else:
                        p=[x for x in Fe if x[0]==i]
                        s=[x[1] for x in Fe if x[0]==i]
                        for y in p:
                            Fe.remove(y)
                        pt.extend(p)
                    t.extend(s)
                    pz.extend(pt)
                T=t
        P=[]
        i=0
        J=[]
        for x in pz:
            if x[0]=='S':
                J.append(i)
            i+=1
        J.append(len(pz))
        for j in range(len(J)-1):
            P.append([pz[x] for x in np.arange(J[j],J[j+1])])
        V=[]
        H=[]
        for x in range(len(P)):
            v=[]
            h=[]
            for y in P[x]:
                if y[0] in Nv and y[1]!='S':
                    v.append(y[0])
                    h.append(y[1])
                elif y[1] in Nv and y[0]!='S':
                    v.append(y[1])
                    h.append(y[0])
            v=list(set(v))
            V.append(v)
            h=list(set(h))
            H.append(h)
        for c in C:
            V[c]=[int(x[1:]) for x in V[c]]
            H[c]=[int(x[1:]) for x in H[c]]
    return V,H,time_r,sol,nods,g


# ## Formulación $\mathcal{F}_3$  con planos cortantes

# In[ ]:


#Hiperaristas contenidas
def F3_pc1(A,W,K):
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
    Nv = ['v{}'.format(i) for i in range(n)]
    Me = ['e{}'.format(i) for i in range(m)]
    w = Nv+Me+['S']
    arcs={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arcx={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arci={(j,i): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    A2 = {**arcs,**arci}
    A1={('S',i): nk for i in Nv}
    arcs.update(arci)
    arcs.update(A1)
    d={}
    for x in w:
        if x in Nv:
            d.update({x:1})
        elif x in Me:
            d.update({x:0})
        else:
            d.update({x:-n})
    arcos, cap = multidict(arcs)
    #Creación del modelo    
    modelo=Model()
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600.0
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    f = modelo.addVars(arcos,vtype=GRB.INTEGER,name='f',lb=0)
    z = modelo.addVars(arcos,vtype=GRB.BINARY, name='z')
    h = modelo.addVars(M,vtype=GRB.BINARY, name='h')
    #Función objetivo
    obj = quicksum(W[j]*h[j] for j in M)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstr(quicksum(z['S',i] for i in Nv) == K) #(2.15)
    modelo.addConstrs(f['S',i] == nk*z['S',i] for i in Nv) #(2.16)
    modelo.addConstrs(z[i,j]<= f[i,j] for (i,j) in A2) #(2.17a)
    modelo.addConstrs(f[i,j] <= (nk-1)*z[i,j] for (i,j) in A2) #(2.17b)
    modelo.addConstrs(f.sum('*',i)-f.sum(i,'*') == d[i] for i in w) #(2.18)
    modelo.addConstrs(z.sum(j,'*') == (sum(A[:,Me.index(j)])-1)*h[Me.index(j)] for j in Me) #(2.19)
    modelo.addConstrs(z.sum('*',j) == h[Me.index(j)] for j in Me) #(2.20)
    modelo.addConstrs(z.sum('*',i) == 1 for i in Nv) #(2.21)
    modelo.addConstrs(z[i,j]+z[j,i]<= 1 for (i,j) in arcx) #(2.22)
    #Hiperaristas contenidas
    h_node = [0 for j in M]
    CONT = cont(A)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for j in M:
                    h_node[j] = modelo.cbGetNodeRel(h[j])
                for p in CONT.keys():
                    for q in CONT[p]:
                        ds = h_node[p]+h_node[q]
                        if  ds.getValue()>1:
                            modelo.cbCut(h[p]+h[q] <= 1)
                            modelo._cbCuts+=1
                        else:
                            pass
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    F = []
    Z = []
    Y = []
    H = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return [],H,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for i in arcos:
            if f[i].x > 0:
                F.append([i,f[i].x])    
        for j in arcos:
            if z[j].x > 0:
                Z.append([j,z[j].x])
                if j[0]=='S':
                    Y.append(j[1])     
        k=len(Y)
        pz=[]
        f=dict(F)
        Fv=[x for x in f if x[0] in Nv]
        Fe=[x for x in f if x[0] in Me]
        for y in Y:
            pz.extend([('S',y)])
            T=[y]
            p=[0]
            pt=[0]
            while len(pt)>0:
                t=[]
                pt = []
                for i in T:
                    if i in Nv:
                        p=[x for x in Fv if x[0]==i]
                        s=[x[1] for x in Fv if x[0]==i]
                        for y in p:
                            Fv.remove(y)
                        pt.extend(p)
                    else:
                        p=[x for x in Fe if x[0]==i]
                        s=[x[1] for x in Fe if x[0]==i]
                        for y in p:
                            Fe.remove(y)
                        pt.extend(p)
                    t.extend(s)
                    pz.extend(pt)
                T=t
        P=[]
        i=0
        J=[]
        for x in pz:
            if x[0]=='S':
                J.append(i)
            i+=1
        J.append(len(pz))
        for j in range(len(J)-1):
            P.append([pz[x] for x in np.arange(J[j],J[j+1])])
        V=[]
        H=[]
        for x in range(len(P)):
            v=[]
            h=[]
            for y in P[x]:
                if y[0] in Nv and y[1]!='S':
                    v.append(y[0])
                    h.append(y[1])
                elif y[1] in Nv and y[0]!='S':
                    v.append(y[1])
                    h.append(y[0])
            v=list(set(v))
            V.append(v)
            h=list(set(h))
            H.append(h)
        for c in C:
            V[c]=[int(x[1:]) for x in V[c]]
            H[c]=[int(x[1:]) for x in H[c]]
    return V,H,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#Número mínimo de hiperaristas
def F3_pc2(A,W,K):
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
    Nv = ['v{}'.format(i) for i in range(n)]
    Me = ['e{}'.format(i) for i in range(m)]
    w = Nv+Me+['S']
    arcs={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arcx={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arci={(j,i): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    A2 = {**arcs,**arci}
    A1={('S',i): nk for i in Nv}
    arcs.update(arci)
    arcs.update(A1)
    d={}
    for x in w:
        if x in Nv:
            d.update({x:1})
        elif x in Me:
            d.update({x:0})
        else:
            d.update({x:-n})
    arcos, cap = multidict(arcs)
    #Creación del modelo    
    modelo=Model()
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600.0
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    f = modelo.addVars(arcos,vtype=GRB.INTEGER,name='f',lb=0)
    z = modelo.addVars(arcos,vtype=GRB.BINARY, name='z')
    h = modelo.addVars(M,vtype=GRB.BINARY, name='h')
    #Función objetivo
    obj = quicksum(W[j]*h[j] for j in M)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstr(quicksum(z['S',i] for i in Nv) == K) #(2.15)
    modelo.addConstrs(f['S',i] == nk*z['S',i] for i in Nv) #(2.16)
    modelo.addConstrs(z[i,j]<= f[i,j] for (i,j) in A2) #(2.17a)
    modelo.addConstrs(f[i,j] <= (nk-1)*z[i,j] for (i,j) in A2) #(2.17b)
    modelo.addConstrs(f.sum('*',i)-f.sum(i,'*') == d[i] for i in w) #(2.18)
    modelo.addConstrs(z.sum(j,'*') == (sum(A[:,Me.index(j)])-1)*h[Me.index(j)] for j in Me) #(2.19)
    modelo.addConstrs(z.sum('*',j) == h[Me.index(j)] for j in Me) #(2.20)
    modelo.addConstrs(z.sum('*',i) == 1 for i in Nv) #(2.21)
    modelo.addConstrs(z[i,j]+z[j,i]<= 1 for (i,j) in arcx) #(2.22)
    #Número mínimo de hiperaristas
    modelo.addConstr(quicksum(h[Me.index(j)] for j in Me)>=K*ALFA(A,nk)[1])
    modelo._cbCuts+=1
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    F = []
    Z = []
    Y = []
    H = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return [],H,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for i in arcos:
            if f[i].x > 0:
                F.append([i,f[i].x])    
        for j in arcos:
            if z[j].x > 0:
                Z.append([j,z[j].x])
                if j[0]=='S':
                    Y.append(j[1])     
        k=len(Y)
        pz=[]
        f=dict(F)
        Fv=[x for x in f if x[0] in Nv]
        Fe=[x for x in f if x[0] in Me]
        for y in Y:
            pz.extend([('S',y)])
            T=[y]
            p=[0]
            pt=[0]
            while len(pt)>0:
                t=[]
                pt = []
                for i in T:
                    if i in Nv:
                        p=[x for x in Fv if x[0]==i]
                        s=[x[1] for x in Fv if x[0]==i]
                        for y in p:
                            Fv.remove(y)
                        pt.extend(p)
                    else:
                        p=[x for x in Fe if x[0]==i]
                        s=[x[1] for x in Fe if x[0]==i]
                        for y in p:
                            Fe.remove(y)
                        pt.extend(p)
                    t.extend(s)
                    pz.extend(pt)
                T=t
        P=[]
        i=0
        J=[]
        for x in pz:
            if x[0]=='S':
                J.append(i)
            i+=1
        J.append(len(pz))
        for j in range(len(J)-1):
            P.append([pz[x] for x in np.arange(J[j],J[j+1])])
        V=[]
        H=[]
        for x in range(len(P)):
            v=[]
            h=[]
            for y in P[x]:
                if y[0] in Nv and y[1]!='S':
                    v.append(y[0])
                    h.append(y[1])
                elif y[1] in Nv and y[0]!='S':
                    v.append(y[1])
                    h.append(y[0])
            v=list(set(v))
            V.append(v)
            h=list(set(h))
            H.append(h)
        for c in C:
            V[c]=[int(x[1:]) for x in V[c]]
            H[c]=[int(x[1:]) for x in H[c]]
    return V,H,time_r,sol,[nods,modelo._cbCuts],g


# In[ ]:


#HM + CO + INT
def F3_pc3(A,W,K):
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
    Nv = ['v{}'.format(i) for i in range(n)]
    Me = ['e{}'.format(i) for i in range(m)]
    w = Nv+Me+['S']
    arcs={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arcx={(i,j): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    arci={(j,i): 1 for i in Nv for j in Me if A[Nv.index(i),Me.index(j)]==1}
    A2 = {**arcs,**arci}
    A1={('S',i): nk for i in Nv}
    arcs.update(arci)
    arcs.update(A1)
    d={}
    for x in w:
        if x in Nv:
            d.update({x:1})
        elif x in Me:
            d.update({x:0})
        else:
            d.update({x:-n})
    arcos, cap = multidict(arcs)
    #Creación del modelo    
    modelo=Model()
    modelo.Params.LogToConsole = 0
    modelo.Params.timeLimit = 3600.0
    modelo.Params.cuts = 0
    modelo._cbCuts = 0
    modelo.Params.PreCrush = 1
    modelo.Params.NodefileStart=0.5
    #creación de variables
    f = modelo.addVars(arcos,vtype=GRB.INTEGER,name='f',lb=0)
    z = modelo.addVars(arcos,vtype=GRB.BINARY, name='z')
    h = modelo.addVars(M,vtype=GRB.BINARY, name='h')
    #Función objetivo
    obj = quicksum(W[j]*h[j] for j in M)
    modelo.setObjective(obj, GRB.MINIMIZE)
    #Restricciones
    modelo.addConstr(quicksum(z['S',i] for i in Nv) == K) #(2.15)
    modelo.addConstrs(f['S',i] == nk*z['S',i] for i in Nv) #(2.16)
    modelo.addConstrs(z[i,j]<= f[i,j] for (i,j) in A2) #(2.17a)
    modelo.addConstrs(f[i,j] <= (nk-1)*z[i,j] for (i,j) in A2) #(2.17b)
    modelo.addConstrs(f.sum('*',i)-f.sum(i,'*') == d[i] for i in w) #(2.18)
    modelo.addConstrs(z.sum(j,'*') == (sum(A[:,Me.index(j)])-1)*h[Me.index(j)] for j in Me) #(2.19)
    modelo.addConstrs(z.sum('*',j) == h[Me.index(j)] for j in Me) #(2.20)
    modelo.addConstrs(z.sum('*',i) == 1 for i in Nv) #(2.21)
    modelo.addConstrs(z[i,j]+z[j,i]<= 1 for (i,j) in arcx) #(2.22)
    #Número mínimo de hiperaristas
    modelo.addConstr(quicksum(h[Me.index(j)] for j in Me)>=K*ALFA(A,nk)[1])
    modelo._cbCuts+=1
    h_node = [0 for j in M]
    CONT = cont(A)
    INT1 = int1(A)
    def mycallback(modelo,where):
        if where == GRB.Callback.MIPNODE:
            if modelo.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for j in M:
                    h_node[j] = modelo.cbGetNodeRel(h[j])
                #Hiperaristas contenidas
                for p in CONT.keys():
                    for q in CONT[p]:
                        ds = h_node[p]+h_node[q]
                        if  ds.getValue()>1:
                            modelo.cbCut(h[p]+h[q] <= 1)
                            modelo._cbCuts+=1
                        else:
                            pass
                #Intersección mayor a 1
                for I in INT1:
                    ds = h_node[I[0]]+h_node[I[1]]
                    if  ds.getValue()>1:
                        modelo.cbCut(h[I[0]]+h[I[1]] <= 1)
                        modelo._cbCuts+=1
    modelo.update()
    modelo.optimize()
    #Recuperación de variables
    F = []
    Z = []
    Y = []
    H = []   
    if modelo.SolCount == 0:
        time_r = modelo.Runtime
        sol = None
        nods = None
        g = None
        return [],H,time_r,sol,nods,g
    else:        
        time_r = modelo.Runtime
        sol = modelo.ObjVal
        nods = modelo.NodeCount
        g = 100*modelo.MIPGap
        for i in arcos:
            if f[i].x > 0:
                F.append([i,f[i].x])    
        for j in arcos:
            if z[j].x > 0:
                Z.append([j,z[j].x])
                if j[0]=='S':
                    Y.append(j[1])     
        k=len(Y)
        pz=[]
        f=dict(F)
        Fv=[x for x in f if x[0] in Nv]
        Fe=[x for x in f if x[0] in Me]
        for y in Y:
            pz.extend([('S',y)])
            T=[y]
            p=[0]
            pt=[0]
            while len(pt)>0:
                t=[]
                pt = []
                for i in T:
                    if i in Nv:
                        p=[x for x in Fv if x[0]==i]
                        s=[x[1] for x in Fv if x[0]==i]
                        for y in p:
                            Fv.remove(y)
                        pt.extend(p)
                    else:
                        p=[x for x in Fe if x[0]==i]
                        s=[x[1] for x in Fe if x[0]==i]
                        for y in p:
                            Fe.remove(y)
                        pt.extend(p)
                    t.extend(s)
                    pz.extend(pt)
                T=t
        P=[]
        i=0
        J=[]
        for x in pz:
            if x[0]=='S':
                J.append(i)
            i+=1
        J.append(len(pz))
        for j in range(len(J)-1):
            P.append([pz[x] for x in np.arange(J[j],J[j+1])])
        V=[]
        H=[]
        for x in range(len(P)):
            v=[]
            h=[]
            for y in P[x]:
                if y[0] in Nv and y[1]!='S':
                    v.append(y[0])
                    h.append(y[1])
                elif y[1] in Nv and y[0]!='S':
                    v.append(y[1])
                    h.append(y[0])
            v=list(set(v))
            V.append(v)
            h=list(set(h))
            H.append(h)
        for c in C:
            V[c]=[int(x[1:]) for x in V[c]]
            H[c]=[int(x[1:]) for x in H[c]]
    return V,H,time_r,sol,[nods,modelo._cbCuts],g


# ## Pruebas

# In[ ]:


P = np.array([[45,100],
              [45,200],
              [45,500],
              [90,150],
              [90,500],
              [90,1000],
              [150,300],
              [150,800],
              [150,1500],
              [180,400],
              [180,1000],
              [180,2000]])
tam = {45:[2,5,10],
       90:[2,5,10,20],
       150:[2,5,10,20],
       180:[2,5,10,20]}
R = []
Rf = []
for p in P:
    for t in tam[p[0]]:
        n = p[0]
        m = p[1]
        K = 3
        sol =None
        tr=0
        i=0
        while sol==None and tr<3000:
            A = rand_hgraf(n,m,t)
            pr = prom(A)
            W = np.random.randint(1,21,size=m)
            if t==2:
                V,E,tr,sol,nod,g = F2(A,W,K)
                V1,E1,tr1,sol1,nod1,g1 = F2_pc1(A,W,K)
                V2,E2,tr2,sol2,nod2,g2 = F2_pc2(A,W,K)
                V3,E3,tr3,sol3,nod3,g3 = F2_pc3(A,W,K)
                Vf,Ef,trf,solf,nodf,gf = F3(A,W,K)
                Vf1,Ef1,trf1,solf1,nodf1,gf1 = None,None,None,None,None,None
                Vf2,Ef2,trf2,solf2,nodf2,gf2 = F3_pc2(A,W,K)
                Vf3,Ef3,trf3,solf3,nodf3,gf3 = F3_pc3(A,W,K)
            else:
                V,E,tr,sol,nod,g = F2(A,W,K)
                V1,E1,tr1,sol1,nod1,g1 = F2_pc1(A,W,K)
                V2,E2,tr2,sol2,nod2,g2 = F2_pc2(A,W,K)
                V3,E3,tr3,sol3,nod3,g3 = F2_pc3(A,W,K)
                Vf,Ef,trf,solf,nodf,gf = F3(A,W,K)
                Vf1,Ef1,trf1,solf1,nodf1,gf1 = F3_pc1(A,W,K)
                Vf2,Ef2,trf2,solf2,nodf2,gf2 = F3_pc2(A,W,K)
                Vf3,Ef3,trf3,solf3,nodf3,gf3 = F3_pc3(A,W,K)
            i += 1
            if i >=10:
                break        
        Tr=[tr,tr1,tr2,tr3]
        S=[sol,sol1,sol2,sol3]
        Nd=[nod,nod1,nod2,nod3]
        G=[g,g1,g2,g3]
        Trf=[trf,trf1,trf2,trf3]
        Sf=[solf,solf1,solf2,solf3]
        Ndf=[nodf,nodf1,nodf2,nodf3]
        Gf=[gf,gf1,gf2,gf3]
        R.append([p,pr,t,Tr,S,Nd,G])
        Rf.append([p,pr,t,Trf,Sf,Ndf,Gf])
        print(p,t)
        print(Tr)
        print(S)
        print(Trf)
        print(Sf)
        print(Nd)
        print(Ndf)


# In[ ]:


Df1 = pd.DataFrame(columns=['Instancia','Prom','Tam. max','Plano cortante','Valor Función Objetivo','GAP', 'Nodos','Tiempo [s]'],
                 index=np.arange(len(R)*6))
pc = ['NA','ES+CO','ES+KS','ES+HM']
for x in range(len(R)):
    for y in range(4):
        Df1.at[4*x+y,'Instancia']=R[x][0]
        Df1.at[4*x+y,'Prom']=R[x][1]
        Df1.at[4*x+y,'Tam. max']=R[x][2]
        Df1.at[4*x+y,'Plano cortante']=pc[y]
        Df1.at[4*x+y,'Valor Función Objetivo']=R[x][4][y]
        Df1.at[4*x+y,'GAP']=R[x][6][y]
        Df1.at[4*x+y,'Nodos']=R[x][5][y]
        Df1.at[4*x+y,'Tiempo [s]']=R[x][3][y]
        
Df2 = pd.DataFrame(columns=['Instancia','Prom','Tam. max','Plano cortante','Valor Función Objetivo','GAP', 'Nodos','Tiempo [s]'],
                 index=np.arange(len(Rf)*5))
pcf = ['NA','CO','HM','HM+CO+INT']
for x in range(len(Rf)):
    for y in range(4):
        Df2.at[4*x+y,'Instancia']=Rf[x][0]
        Df2.at[4*x+y,'Prom']=Rf[x][1]
        Df2.at[4*x+y,'Tam. max']=Rf[x][2]
        Df2.at[4*x+y,'Plano cortante']=pcf[y]
        Df2.at[4*x+y,'Valor Función Objetivo']=Rf[x][4][y]
        Df2.at[4*x+y,'GAP']=Rf[x][6][y]
        Df2.at[4*x+y,'Nodos']=Rf[x][5][y]
        Df2.at[4*x+y,'Tiempo [s]']=Rf[x][3][y]


# In[ ]:


Df1.to_excel('pruebas_F2.xlsx')
Df2.to_excel('pruebas_F3.xlsx')


# In[ ]:


df = pd.read_excel('pruebas_F2.xlsx') 
INST = df['Instancia'].tolist()
Tiempo = df['Tiempo [s]'].tolist()
Plano = df['Plano cortante'].tolist()
GAP = df['GAP'].tolist()
TAM = df['Tam. max'].tolist()
P = list([[45,100],
          [45,200],
          [45,500],
          [90,150],
          [90,500],
          [90,1000],
          [150,300],
          [150,800],
          [150,1500],
          [180,400],
          [180,1000],
          [180,2000]])
pc = ['NA','ES+CO','ES+KS','ES+HM']


# In[ ]:


I = []
for x in P:
    if x[0] == 45:
        for y in [2,5,10]:
            I.append([x,y])
    elif x[0] == 90:
        for y in [2,5,10,20]:
            I.append([x,y])
    elif x[0] == 150:
        for y in [2,5,10,20]:
            I.append([x,y])
    elif x[0] == 180:
        for y in [2,5,10,20]:
            I.append([x,y])


# In[ ]:


T_tam2 = []
T_tam5  = []
T_tam10  = []
T_tam20  = []
G_tam2  = []
G_tam5  = []
G_tam10  = []
G_tam20  = []
com2=[]
com5=[]
com10=[]
com20=[]
for x in range(len(I)):
    tam = I[x][1]
    if tam==2:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam2.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam2.append(g)
        com2.append(str(I[x][0]))
    elif tam==5:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam5.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam5.append(g)
        com5.append(str(I[x][0]))
    elif tam==10:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam10.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam10.append(g)
        com10.append(str(I[x][0]))
    elif tam==20:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam20.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam20.append(g)
        com20.append(str(I[x][0]))


# In[ ]:


fig1 ,axs1 = plt.subplots(1,2,figsize=(26,12)) #tam2
fig2 ,axs2 = plt.subplots(1,2,figsize=(26,12)) #tam5
fig3 ,axs3 = plt.subplots(1,2,figsize=(26,12)) #tam10
fig4 ,axs4 = plt.subplots(1,2,figsize=(26,12)) #tam20
x_offset=0.5
import matplotlib.lines as mlines
line_parm = {
            0:{'color':'red', 'marker':'o', 'linestyle':'dashed','linewidth':2, 'markersize':5, 'label':'Ninguno'},
            1:{'color':'green', 'marker':'x', 'linestyle':'-', 'linewidth':1,'markersize':5, 'label':'ES+CO'},
            2:{'color':'blue', 'marker':'v', 'linestyle':'-', 'linewidth':1,'markersize':5, 'label':'ES+KS'},
            3:{'color':'cyan', 'marker':'s', 'linestyle':'-', 'linewidth':1,'markersize':5, 'label':'ES+HM'}
}
mod = [
    mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='Ninguno'),
     mlines.Line2D([], [], color='green', marker='x', linestyle='None',
                        markersize=5, label='ES+CO'),
     mlines.Line2D([], [], color='blue', marker='v', linestyle='None',
                        markersize=5, label='ES+KS'),
     mlines.Line2D([], [], color='cyan', marker='s', linestyle='None',
                        markersize=5, label='ES+HM')]
#TAM2
axs1[0].set_xticklabels(com2,rotation=90,fontsize=20)
axs1[0].tick_params(axis='y', labelsize=15)
axs1[0].set_xlabel('[V E]',fontsize=15)
axs1[0].set_yscale('log')
axs1[0].set_ylabel('tiempo [s]',fontsize=15)
axs1[0].plot(com2,[x[0] for x in T_tam2],**line_parm[0])
axs1[0].plot(com2,[x[1] for x in T_tam2],**line_parm[1])
axs1[0].plot(com2,[x[2] for x in T_tam2],**line_parm[2])
axs1[0].plot(com2,[x[3] for x in T_tam2],**line_parm[3])
axs1[0].legend(handles=mod,fontsize=15)
axs1[0].grid(True)
axs1[1].set_xticklabels(com2,rotation=90,fontsize=20)
axs1[1].tick_params(axis='y', labelsize=15)
axs1[1].set_xlabel('[V E]',fontsize=15)
axs1[1].set_ylabel('GAP [%]',fontsize=15)
axs1[1].set_ylim(-0.1)
axs1[1].plot(com2,[x[0] for x in G_tam2],**line_parm[0])
axs1[1].plot(com2,[x[1] for x in G_tam2],**line_parm[1])
axs1[1].plot(com2,[x[2] for x in G_tam2],**line_parm[2])
axs1[1].plot(com2,[x[3] for x in G_tam2],**line_parm[3])
axs1[1].legend(handles=mod,fontsize=15)
axs1[1].grid(True)
fig1.savefig('F2_tam2.jpg',bbox_inches='tight',dpi=300)
#TAM5
axs2[0].set_xticklabels(com5,rotation=90,fontsize=20)
axs2[0].tick_params(axis='y', labelsize=15)
axs2[0].set_xlabel('[V E]',fontsize=15)
axs2[0].set_yscale('log')
axs2[0].set_ylabel('tiempo [s]',fontsize=15)
axs2[0].plot(com5,[x[0] for x in T_tam5],**line_parm[0])
axs2[0].plot(com5,[x[1] for x in T_tam5],**line_parm[1])
axs2[0].plot(com5,[x[2] for x in T_tam5],**line_parm[2])
axs2[0].plot(com5,[x[3] for x in T_tam5],**line_parm[3])
axs2[0].legend(handles=mod,fontsize=15)
axs2[0].grid(True)
axs2[1].set_xticklabels(com5,rotation=90,fontsize=20)
axs2[1].tick_params(axis='y', labelsize=15)
axs2[1].set_xlabel('[V E]',fontsize=15)
axs2[1].set_ylabel('GAP [%]',fontsize=15)
axs2[1].set_ylim(-0.1)
axs2[1].plot(com5,[x[0] for x in G_tam5],**line_parm[0])
axs2[1].plot(com5,[x[1] for x in G_tam5],**line_parm[1])
axs2[1].plot(com5,[x[2] for x in G_tam5],**line_parm[2])
axs2[1].plot(com5,[x[3] for x in G_tam5],**line_parm[3])
axs2[1].legend(handles=mod,fontsize=15)
axs2[1].grid(True)
fig2.savefig('F2_tam5.jpg',bbox_inches='tight',dpi=300)
#TAM10
axs3[0].set_xticklabels(com10,rotation=90,fontsize=20)
axs3[0].tick_params(axis='y', labelsize=15)
axs3[0].set_xlabel('[V E]',fontsize=15)
axs3[0].set_yscale('log')
axs3[0].set_ylabel('tiempo [s]',fontsize=15)
axs3[0].plot(com10,[x[0] for x in T_tam10],**line_parm[0])
axs3[0].plot(com10,[x[1] for x in T_tam10],**line_parm[1])
axs3[0].plot(com10,[x[2] for x in T_tam10],**line_parm[2])
axs3[0].plot(com10,[x[3] for x in T_tam10],**line_parm[3])
axs3[0].legend(handles=mod,fontsize=15)
axs3[0].grid(True)
axs3[1].set_xticklabels(com10,rotation=90,fontsize=20)
axs3[1].tick_params(axis='y', labelsize=15)
axs3[1].set_xlabel('[V E]',fontsize=15)
axs3[1].set_xlim(-x_offset,len(com10)-1+x_offset)
axs3[1].set_ylabel('GAP [%]',fontsize=15)
axs3[1].set_ylim(-5,85)
axs3[1].plot(com10,[x[0] for x in G_tam10],**line_parm[0])
axs3[1].plot(com10,[x[1] for x in G_tam10],**line_parm[1])
axs3[1].plot(com10,[x[2] for x in G_tam10],**line_parm[2])
axs3[1].plot(com10,[x[3] for x in G_tam10],**line_parm[3])
axs3[1].legend(handles=mod,fontsize=15)
axs3[1].grid(True)
fig3.savefig('F2_tam10.jpg',bbox_inches='tight',dpi=300)
#TAM20
axs4[0].set_xticklabels(com20,rotation=90,fontsize=20)
axs4[0].tick_params(axis='y', labelsize=15)
axs4[0].set_xlabel('[V E]',fontsize=15)
axs4[0].set_yscale('log')
axs4[0].set_ylabel('tiempo [s]',fontsize=15)
axs4[0].plot(com20,[x[0] for x in T_tam20],**line_parm[0])
axs4[0].plot(com20,[x[1] for x in T_tam20],**line_parm[1])
axs4[0].plot(com20,[x[2] for x in T_tam20],**line_parm[2])
axs4[0].plot(com20,[x[3] for x in T_tam20],**line_parm[3])
axs4[0].legend(handles=mod,fontsize=15)
axs4[0].grid(True)
axs4[1].set_xticklabels(com20,rotation=90,fontsize=20)
axs4[1].tick_params(axis='y', labelsize=15)
axs4[1].set_xlabel('[V E]',fontsize=15)
axs4[1].set_xlim(-x_offset,len(com20)-1+x_offset)
axs4[1].set_ylabel('GAP [%]',fontsize=15)
axs4[1].set_ylim(-5,105)
axs4[1].plot(com20,[x[0] for x in G_tam20],**line_parm[0])
axs4[1].plot(com20,[x[1] for x in G_tam20],**line_parm[1])
axs4[1].plot(com20,[x[2] for x in G_tam20],**line_parm[2])
axs4[1].plot(com20,[x[3] for x in G_tam20],**line_parm[3])
axs4[1].legend(handles=mod,fontsize=15)
axs4[1].grid(True)
fig4.savefig('F2_tam20.jpg',bbox_inches='tight',dpi=300)
plt.show()


# In[ ]:


df = pd.read_excel('pruebas_F3.xlsx') 
INST = df['Instancia'].tolist()
Tiempo = df['Tiempo [s]'].tolist()
Plano = df['Plano cortante'].tolist()
GAP = df['GAP'].tolist()
TAM = df['Tam. max'].tolist()
P = list([[45,100],
          [45,200],
          [45,500],
          [90,150],
          [90,500],
          [90,1000],
          [150,300],
          [150,800],
          [150,1500],
          [180,400],
          [180,1000],
          [180,2000]])
pcf = ['NA','CO','HM','HM+CO+INT']


# In[ ]:


I = []
for x in P:
    if x[0] == 45:
        for y in [2,5,10]:
            I.append([x,y])
    elif x[0] == 90:
        for y in [2,5,10,20]:
            I.append([x,y])
    elif x[0] == 150:
        for y in [2,5,10,20]:
            I.append([x,y])
    elif x[0] == 180:
        for y in [2,5,10,20]:
            I.append([x,y])


# In[ ]:


T_tam2 = []
T_tam5  = []
T_tam10  = []
T_tam20  = []
G_tam2  = []
G_tam5  = []
G_tam10  = []
G_tam20  = []
com2=[]
com5=[]
com10=[]
com20=[]
for x in range(len(I)):
    tam = I[x][1]
    if tam==2:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam2.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam2.append(g)
        com2.append(str(I[x][0]))
    elif tam==5:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam5.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam5.append(g)
        com5.append(str(I[x][0]))
    elif tam==10:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam10.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam10.append(g)
        com10.append(str(I[x][0]))
    elif tam==20:
        tiempo = [Tiempo[4*x+y] for y in range(4)]
        T_tam20.append(tiempo)
        g = [GAP[4*x+y] for y in range(4)]
        G_tam20.append(g)
        com20.append(str(I[x][0]))


# In[ ]:


fig1 ,axs1 = plt.subplots(1,2,figsize=(26,12)) #tam2
fig2 ,axs2 = plt.subplots(1,2,figsize=(26,12)) #tam5
fig3 ,axs3 = plt.subplots(1,2,figsize=(26,12)) #tam10
fig4 ,axs4 = plt.subplots(1,2,figsize=(26,12)) #tam20
x_offset=0.5
import matplotlib.lines as mlines
line_parm = {
            0:{'color':'red', 'marker':'o', 'linestyle':'dashed','linewidth':2, 'markersize':5, 'label':'Ninguno'},
            1:{'color':'green', 'marker':'x', 'linestyle':'-', 'linewidth':1,'markersize':5, 'label':'CO'},
            2:{'color':'blue', 'marker':'v', 'linestyle':'-', 'linewidth':1,'markersize':5, 'label':'HM'},
            3:{'color':'cyan', 'marker':'s', 'linestyle':'-', 'linewidth':1,'markersize':5, 'label':'HM+CO+INT'}
}
mod = [
    mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=5, label='Ninguno'),
     mlines.Line2D([], [], color='green', marker='x', linestyle='None',
                        markersize=5, label='CO'),
     mlines.Line2D([], [], color='blue', marker='v', linestyle='None',
                        markersize=5, label='HM'),
     mlines.Line2D([], [], color='cyan', marker='s', linestyle='None',
                        markersize=5, label='HM+CO+INT')]
#TAM2
axs1[0].set_xticklabels(com2,rotation=90,fontsize=20)
axs1[0].tick_params(axis='y', labelsize=15)
axs1[0].set_xlabel('[V E]',fontsize=15)
axs1[0].set_yscale('log')
axs1[0].set_ylabel('tiempo [s]',fontsize=15)
axs1[0].plot(com2,[x[0] for x in T_tam2],**line_parm[0])
axs1[0].plot(com2,[x[1] for x in T_tam2],**line_parm[1])
axs1[0].plot(com2,[x[2] for x in T_tam2],**line_parm[2])
axs1[0].plot(com2,[x[3] for x in T_tam2],**line_parm[3])
axs1[0].legend(handles=mod,fontsize=15)
axs1[0].grid(True)
axs1[1].set_xticklabels(com2,rotation=90,fontsize=20)
axs1[1].tick_params(axis='y', labelsize=15)
axs1[1].set_xlabel('[V E]',fontsize=15)
axs1[1].set_ylabel('GAP [%]',fontsize=15)
axs1[1].set_ylim(-0.1)
axs1[1].plot(com2,[x[0] for x in G_tam2],**line_parm[0])
axs1[1].plot(com2,[x[1] for x in G_tam2],**line_parm[1])
axs1[1].plot(com2,[x[2] for x in G_tam2],**line_parm[2])
axs1[1].plot(com2,[x[3] for x in G_tam2],**line_parm[3])
axs1[1].legend(handles=mod,fontsize=15)
axs1[1].grid(True)
fig1.savefig('F3_tam2.jpg',bbox_inches='tight',dpi=300)
#TAM5
axs2[0].set_xticklabels(com5,rotation=90,fontsize=20)
axs2[0].tick_params(axis='y', labelsize=15)
axs2[0].set_xlabel('[V E]',fontsize=15)
axs2[0].set_yscale('log')
axs2[0].set_ylabel('tiempo [s]',fontsize=15)
axs2[0].plot(com5,[x[0] for x in T_tam5],**line_parm[0])
axs2[0].plot(com5,[x[1] for x in T_tam5],**line_parm[1])
axs2[0].plot(com5,[x[2] for x in T_tam5],**line_parm[2])
axs2[0].plot(com5,[x[3] for x in T_tam5],**line_parm[3])
axs2[0].legend(handles=mod,fontsize=15)
axs2[0].grid(True)
axs2[1].set_xticklabels(com5,rotation=90,fontsize=20)
axs2[1].tick_params(axis='y', labelsize=15)
axs2[1].set_xlabel('[V E]',fontsize=15)
axs2[1].set_ylabel('GAP [%]',fontsize=15)
axs2[1].set_ylim(-0.1)
axs2[1].plot(com5,[x[0] for x in G_tam5],**line_parm[0])
axs2[1].plot(com5,[x[1] for x in G_tam5],**line_parm[1])
axs2[1].plot(com5,[x[2] for x in G_tam5],**line_parm[2])
axs2[1].plot(com5,[x[3] for x in G_tam5],**line_parm[3])
axs2[1].legend(handles=mod,fontsize=15)
axs2[1].grid(True)
fig2.savefig('F3_tam5.jpg',bbox_inches='tight',dpi=300)
#TAM10
axs3[0].set_xticklabels(com10,rotation=90,fontsize=20)
axs3[0].tick_params(axis='y', labelsize=15)
axs3[0].set_xlabel('[V E]',fontsize=15)
axs3[0].set_yscale('log')
axs3[0].set_ylabel('tiempo [s]',fontsize=15)
axs3[0].plot(com10,[x[0] for x in T_tam10],**line_parm[0])
axs3[0].plot(com10,[x[1] for x in T_tam10],**line_parm[1])
axs3[0].plot(com10,[x[2] for x in T_tam10],**line_parm[2])
axs3[0].plot(com10,[x[3] for x in T_tam10],**line_parm[3])
axs3[0].legend(handles=mod,fontsize=15)
axs3[0].grid(True)
axs3[1].set_xticklabels(com10,rotation=90,fontsize=20)
axs3[1].tick_params(axis='y', labelsize=15)
axs3[1].set_xlabel('[V E]',fontsize=15)
axs3[1].set_xlim(-x_offset,len(com10)-1+x_offset)
axs3[1].set_ylabel('GAP [%]',fontsize=15)
axs3[1].set_ylim(-5,85)
axs3[1].plot(com10,[x[0] for x in G_tam10],**line_parm[0])
axs3[1].plot(com10,[x[1] for x in G_tam10],**line_parm[1])
axs3[1].plot(com10,[x[2] for x in G_tam10],**line_parm[2])
axs3[1].plot(com10,[x[3] for x in G_tam10],**line_parm[3])
axs3[1].legend(handles=mod,fontsize=15)
axs3[1].grid(True)
fig3.savefig('F3_tam10.jpg',bbox_inches='tight',dpi=300)
#TAM20
axs4[0].set_xticklabels(com20,rotation=90,fontsize=20)
axs4[0].tick_params(axis='y', labelsize=15)
axs4[0].set_xlabel('[V E]',fontsize=15)
axs4[0].set_yscale('log')
axs4[0].set_ylabel('tiempo [s]',fontsize=15)
axs4[0].plot(com20,[x[0] for x in T_tam20],**line_parm[0])
axs4[0].plot(com20,[x[1] for x in T_tam20],**line_parm[1])
axs4[0].plot(com20,[x[2] for x in T_tam20],**line_parm[2])
axs4[0].plot(com20,[x[3] for x in T_tam20],**line_parm[3])
axs4[0].legend(handles=mod,fontsize=15)
axs4[0].grid(True)
axs4[1].set_xticklabels(com20,rotation=90,fontsize=20)
axs4[1].tick_params(axis='y', labelsize=15)
axs4[1].set_xlabel('[V E]',fontsize=15)
axs4[1].set_xlim(-x_offset,len(com20)-1+x_offset)
axs4[1].set_ylabel('GAP [%]',fontsize=15)
axs4[1].set_ylim(-5,105)
axs4[1].plot(com20,[x[0] for x in G_tam20],**line_parm[0])
axs4[1].plot(com20,[x[1] for x in G_tam20],**line_parm[1])
axs4[1].plot(com20,[x[2] for x in G_tam20],**line_parm[2])
axs4[1].plot(com20,[x[3] for x in G_tam20],**line_parm[3])
axs4[1].legend(handles=mod,fontsize=15)
axs4[1].grid(True)
fig4.savefig('F3_tam20.jpg',bbox_inches='tight',dpi=300)
plt.show()


# In[ ]:





# In[ ]:




