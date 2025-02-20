# k-way Hypergraph Equipartitioning. Guillermo Landazuri, Diego Recalde, Ramiro Torres, Polo Vaca.

## Files and Folders in this repository

*['ReadMe.md](ReadMe.md): This file.

*['Instances'](Instances): Contains all of the instances used in this work. Instance name is identified by  n_m_me_k.txt, where n=number of nodes, m=number of hyperedges, me=hyperedge maximum cardinality, and k= number of components in which the Hypergraph must be partitioned. Moreover, if A represents the incidence matrix of the hypergraph and w represents the positive cost on the hyperedges, the format of each instance is:
n m k 
w
A 

*['Formulation F1'](F1_2024_cuts.py): Code in Python for k-way Hypergraph Equipartitioning Problem (Formulation F1). The Gurobi package is needed for the linear models.

*['Formulation F2 and F3'](F2_F3_2024_cuts.py): Code in Python for k-way Hypergraph Equipartitioning in Linear Hyper-trees (Formulation F2 and F3). The Gurobi package is needed for the linear models.

