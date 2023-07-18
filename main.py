
# coding: utf-8

# In[ ]:


import numpy as np
import mdptoolbox, mdptoolbox.example
import random

s = 10
a = 5
discount = 0.6

episodes = 100
iterations = 100000

ql_total_diff = np.zeros((episodes,iterations))
sql_total_diff= np.zeros((episodes,iterations))

ql_avg_diff_per_episode = np.zeros((episodes))
sql_avg_diff_per_episode = np.zeros((episodes))
# gsql1_total_diff= np.zeros((episodes,iterations))
# gsql2_total_diff=np.zeros((episodes,iterations))

for count in range(episodes):
    print(count)
    np.random.seed((count+1)*100)
    random.seed((count+1)*110)
    
    P, R = mdptoolbox.example.rand(s, a)
    
    vi = mdptoolbox.mdp.ValueIteration(P, R, discount,epsilon=0.00001)
    vi.run()

    #Q-Learning
    qlearn = mdptoolbox.mdp.QLearning(P, R, discount, n_iter=iterations)
    qlearn.run()
    # print("vi.V shape :",len(vi.V)," qlearn.q_values shape ",qlearn.q_values.shape)
    norm_diff_ql = vi.V - qlearn.q_values
    ql_total_diff[count] = np.linalg.norm(norm_diff_ql,axis =1)
    # ql_avg_diff_per_episode[count] = np.mean(ql_total_diff[count])

    #Speedy QL
    ql1 = mdptoolbox.mdp.SpeedyQLearning(P, R, discount,n_iter=iterations)
    ql1.run()
    # print("vi.V shape :",len(vi.V)," sqlearn.q_values shape ",qlearn.q_values.shape)
    norm_diff_sql = vi.V - ql1.q_values
    sql_total_diff[count] = np.linalg.norm(norm_diff_sql,axis =1)
    sql_avg_diff_per_episode[count] = np.mean(sql_total_diff[count])
    
    #GSQL1
    # ql2 = mdptoolbox.mdp.GSQL1(P, R, discount,n_iter=iterations)
    # ql2.run()
    # norm_diff_gsql1 = vi.V - ql2.q_values
    # gsql1_total_diff[count] = np.linalg.norm(norm_diff_gsql1,axis =1)
    
    # #GSQL2
    # ql3 = mdptoolbox.mdp.GSQL2(P, R, discount,n_iter=iterations)
    # ql3.run()
    # norm_diff_gsql2 = vi.V - ql3.q_values
    # gsql2_total_diff[count] = np.linalg.norm(norm_diff_gsql2,axis =1)

#Calculate average error and standard deviation
avg_error_ql=np.mean(ql_total_diff,axis=0)
std_error_ql=np.std(ql_total_diff,axis=0)   
avg_error_sql=np.mean(sql_total_diff,axis=0)
std_error_sql=np.std(sql_total_diff,axis=0)
# avg_error_gsql1=np.mean(gsql1_total_diff,axis=0)
# std_error_gsql1=np.std(gsql1_total_diff,axis=0)
# avg_error_gsql2=np.mean(gsql2_total_diff,axis=0)
# std_error_gsql2=np.std(gsql2_total_diff,axis=0)
# np.savetxt("mean_std.csv", np.c_[avg_error_sql,avg_error_gsql1,avg_error_gsql2,std_error_sql,std_error_gsql1,std_error_gsql2], delimiter=",")
np.savetxt("mean_std.csv", np.c_[avg_error_ql,avg_error_sql,std_error_ql,std_error_sql], delimiter=",")

