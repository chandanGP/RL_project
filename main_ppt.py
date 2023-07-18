ql_total_diff = np.zeros((num_models,iterations))
sql_total_diff= np.zeros((num_models,iterations))
for count in range(num_models):
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
    norm_diff_ql = vi.V - qlearn.V_for_n_iter #shape V = |S|, V_for_n_iter = (iterations,|S|)
    ql_total_diff[count] = np.linalg.norm(norm_diff_ql,axis =1) #norm difference for each iteration


    #Speedy QL
    ql1 = mdptoolbox.mdp.SpeedyQLearning(P, R, discount,n_iter=iterations)
    ql1.run()
    # print("vi.V shape :",len(vi.V)," sqlearn.q_values shape ",qlearn.q_values.shape)
    norm_diff_sql = vi.V - ql1.V_for_n_iter #shape V = |S|, V_for_n_iter = (iterations,|S|)
    sql_total_diff[count] = np.linalg.norm(norm_diff_sql,axis =1) #norm difference for each iteration

    

#Calculate average error and standard deviation for models at each iteration
avg_error_ql=np.mean(ql_total_diff,axis=0)
std_error_ql=np.std(ql_total_diff,axis=0)   
avg_error_sql=np.mean(sql_total_diff,axis=0)
std_error_sql=np.std(sql_total_diff,axis=0)