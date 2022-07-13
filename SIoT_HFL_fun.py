import networkx as nx
import numpy as np
import random
from random import randint

num_node = 200
node_degree = 10
computation_mean = 6
computation_std = 2
communication_mean = 6
communication_std = 2
public_ratio = 0.1
cc_cost = 10      #communication cost of connected cloud

# user info
num_user =int(num_node/30)
trust_mean = 0.75
trust_std = 0.3
u_s_table = {}   # user-to-SIoT table
s_u_table = {}   # SIoT-to-user table
users_trust=[]   # trust table

# data distribution
label = 10
IID = False
primary = 0.6
data_mean = 300
data_std_ratio = 0.3    #need to less than 0.45
label_std_ratio = 0.1   #need to less than 0.45

# hiring cost
u_data = 0.01   # unit data cost
hired_SIoTs =set()

# privacy setting
sigma = 0.25      # adjustable parameter ~[0,1]
epsilon = 15      # privacy intensity parameter
p_cost = 30       # privacy cost

# data coverage threshold
B = 2000
l_num_data = [B]*label
total_num_data = [0]*label

# data balancing threshold
D = 800

#cluster info 
clusters=[]

def rand_ints(mean, std, size):
    randomNums = np.random.normal(loc=mean, scale=std, size=size)
    randomInts = np.round(randomNums)
    randomInts = list(randomInts)
    f_randomInts = [int(x) if x>=2 else 1 for x in randomInts]
    return f_randomInts

def rand_data (mean, std, size):
    randomNums = np.random.normal(loc=mean, scale=std, size=size)
    randomInts = np.round(randomNums)
    randomInts = list(randomInts)
    lower = int(mean-(std*2))
    d_randomInts = [int(x) if x>=lower else lower for x in randomInts]
    return d_randomInts

def rand_trust(mean, std, size):
    randomNums = np.random.normal(loc=mean, scale=std, size=size)
    randomNums = list(randomNums)
    t_random = []
    for x in randomNums:
        
        if x >=0.99:
            t_random.append(0.97)
        elif x < (mean-std):
            t_random.append(mean-std)
        else:
            t_random.append(x)
    
    return t_random



#-----------generate SIoT node and communication topology----------------
def generated_node(size, degree, cmp_mean, cmp_std, 
                   cmu_mean, cmu_std, 
                   lab, iid, primary, num_data, d_std_ratio, l_std_ratio):
    
#-----------generate SIoT node with computation cost and data -------------
    SIoT_nodes = []
    # generate the computation cost
    computation_costs = rand_ints(cmp_mean, cmp_std, size)
    data_distribution = rand_data(num_data, num_data*d_std_ratio, size)
    print("total data:",sum(data_distribution))
#     print(data_distribution)
    # generate the node with the computation cost and the data distribution
    #iid setting
    if iid:
        print ("Data distribution is IID.")
        for x, y in enumerate(computation_costs):
        
            n_data = data_distribution[x]
            l_std = l_std_ratio*n_data/lab
        
            # generate the label distribution
            label_num = rand_ints(n_data/lab, l_std, lab)
            
            #combine the computation cost and data distribution
            node_info = (x, {"cmp":y,
                            "total_data": sum(label_num),
                            "d_data": label_num,
                            "p_level": 0,      # the privacy requirment 
                            "pri_n": -1})    # the node that let it has this p_level 
            SIoT_nodes.append(node_info)
    
    else:
        print ("Data distribution is non-IID.")
        for x, y in enumerate(computation_costs):
            
            n_data = data_distribution[x]
    
            i = randint(0, lab-1)
            label_num = rand_ints(n_data*(1-primary)/9, l_std_ratio*n_data*(1-primary)/9, lab)
            label_num[i] = int (n_data*primary)
            node_info = (x, {"cmp":y,
                            "total_data": sum(label_num),
                            "d_data": label_num,
                            "p_level": 0,      # the privacy requirment 
                            "pri_n": -1})    # the node that let it has this p_level 
            SIoT_nodes.append(node_info)

#-----------generate SIoT edge with communication cost -------------    
    
    edges = []
    # generate edges
    for x in range(num_node):
        e = []
        while(len(e)< node_degree):
            i = randint(0, num_node-1)
            if (i not in e and i!= x):
                e.append(i)
        for j in e:
            edges.append((x,j))

    # remove duplicate edges  EX. (a,b), (b,a)  
    for e in edges:
        if ((e[1], e[0]) in edges):
            edges.remove((e[1], e[0]))

    #generate the communication cost
    communication_costs = rand_ints(cmu_mean, cmu_std, len(edges))

    #combine the edges and the communication cost
    edges_cmu = [(edges[x][0], edges[x][1], {"cmu":y}) for x, y in enumerate(communication_costs)]
    
    return SIoT_nodes, edges_cmu



def assign_SIoT_to_user(node, user, ratio):   
    avg_node = int((1-ratio)*node/user)
    nodes = list(range(node))
    node_distribution = {}
    user_distribution = {}
    
    # assign SIoT to user
    for x in range(user):
        assign_nodes = sorted(random.sample(nodes, avg_node))
        node_distribution[x] = assign_nodes

        # remove the assigned nodes
        for i in assign_nodes:
            nodes.remove(i)

    # assign the remaining nodes
    node_distribution["public"] = nodes
    
    
    #construct the SIoT-to-user dict.
    for u, SIoTs in node_distribution.items():
        for s in SIoTs:
            user_distribution[s] = u
            
    return node_distribution, user_distribution


#-----------generate the trust table of the users -------------
def con_trust_table(n_user):
    trust_table = [[0]*n_user for i in range(n_user)]
    num_trust = int(n_user*(n_user-1)/2)
    trusts = rand_trust(trust_mean, trust_std, num_trust)
    for i in range(0, n_user):
        trust_table[i][i] = 1.0
    k=0
    for i in range(0, n_user):
        for j in range(i+1, n_user):
            trust_table[i][j] = float(format(trusts[k], '.2f'))
            trust_table[j][i] = float(format(trusts[k], '.2f'))
            k+=1
    return trust_table


# ---------Step1. SIoT Selection and Clustering ~~~ Function ------------ 


# ---------compute the vaild number of labels ------------
def cmp_vaild_data(data):
    v_n_data = 0    # vaild the number of data 
    for i in range(0, label):
        if ((l_num_data[i] - data[i])>=0):
            v_n_data += data[i]
        else:
            v_n_data += l_num_data[i]

    return v_n_data


# ---------updata the lacking number of labels ------------
def update_l_data(data):
    print("Current the lacking number of labels  :",l_num_data)
    print("Current the removing number of labels :",data)
    for i in range(0, label):
        total_num_data[i] += data[i]
        if ((l_num_data[i] - data[i])>=0):
            l_num_data[i] -= data[i]
        else:
            l_num_data[i] =0

    print("The lacking number of labels after update:",l_num_data)
    print("----------------total data---------------:",total_num_data)
    

# --------- compute trust-to-privacy, privacy-to-cost------------
def trust_to_privacy(t):
    return float(format(epsilon*t/(t+sigma), '.2f'))

def trust_to_cost(t):
    return float(format((1-trust_to_privacy(t)/epsilon)*p_cost, '.2f'))

def privacy_to_cost(p):
    return float(format((1-p/epsilon)*p_cost, '.2f'))



# --------- get the user who own this SIoT ------------
def get_user(i):
    return s_u_table[i]


# --------- get the trust betweent two SIoT ------------
def get_2SIoT_trust(i,j):
    
    u1 = s_u_table[i]
    u2 = s_u_table[j]
    
    if (u1=='public') or (u2=='public'):
        return 1.0
    else: 
        return users_trust[u1][u2]
    
    
    
# --------- create new cluster info. ------------    
def create_cluster(p):
    
    G = nx.Graph()
    n = [(p[0], G_s.nodes[p[0]]),(p[1], G_s.nodes[p[1]])]
    e = [(p[0], p[1], G_s[p[0]][p[1]])]
    G.add_nodes_from(n)
    G.add_edges_from(e)
    cluster_info={
        'id': len(clusters),
        'graph': G,
        'node': [p[0], p[1]],
        'p_level': get_2SIoT_trust(p[0], p[1]),
        'CH': None,
        'total_cmp':0,
        'total_cmu':0,
        'total_hir':0,
        'total_pri':0,
        'total_cst':0
        
    }
    clusters.append(cluster_info)    
    
# --------- find the candidate pair ------------    
def find_cnd_c(ps):
    cnd_c = None
    CP_c = 0
    for p in ps :
        cmp_cost = G_s.nodes[p[0]]['cmp']+G_s.nodes[p[1]]['cmp']
        cmu_cost = G_s[p[0]][p[1]]['cmu']
        hir_cost = (G_s.nodes[p[0]]['total_data']+G_s.nodes[p[1]]['total_data'])*u_data

        t = get_2SIoT_trust(p[0],p[1])
        pri_cost = 2*trust_to_cost(t)
        t_cost = cmp_cost + cmu_cost + hir_cost + pri_cost + cc_cost
#          print ("cmp_cost:{}, cmu_cost: {}, hir_cost:{}, pri_cost:{} and t_cost:{}"
#                .format(cmp_cost, cmu_cost, hir_cost, pri_cost, t_cost))


        t_dataset = [G_s.nodes[p[0]]['d_data'][i]+G_s.nodes[p[1]]['d_data'][i] for i in range(0, label)]
        v_data = cmp_vaild_data(t_dataset)
#         print("t_dataset:{}, v_data:{}".format(t_dataset, v_data))

        CP = v_data/t_cost
#         print ("CP is ", CP )
        
        if (CP >= CP_c):
            cnd_c = p
            CP_c = CP
    return cnd_c, CP_c



# --------- find the min cmu edge of SIoT i in cluster c------------ 
def get_min_cmu_edge(i,c):
    i_nbr = set(G_s.neighbors(i))
    i_nbr = i_nbr & set(c)
    cmu = 10000
    n = None
    for j in i_nbr:
        if (cmu>G_s[i][j]['cmu']):
            cmu = G_s[i][j]['cmu']
            n = j
    return n



# --------- find the min trust of SIoT i and cluster c------------
def get_min_trust(i,c, min_trust):
    min_t = min_trust
    for j in c:
        if (min_t > get_2SIoT_trust(i, j)):
            min_t = get_2SIoT_trust(i, j)
    return min_t



#---------- find the candidate SIoT of each cluster --------------
def find_c_SIoTs(cs):
    cnd_SIoTs = []
    for c in cs:
        adj_c =set()
        for n in c['node']:
            adj_c = adj_c|set(G_s.neighbors(n))
        adj_c = adj_c - hired_SIoTs
    #     print(adj_c)

        cnd_s = None
        CP_s = 0
        nbr_min_c = None
        min_trust = 0
        for nbr in adj_c:       
            cmp_cost = G_s.nodes[nbr]['cmp']

            # find the SIoT that has min cmu edge connected the SIoT nbr in cluster c 
            nbr_min = get_min_cmu_edge(nbr,c['node'])
            cmu_cost = G_s[nbr][nbr_min]['cmu']
            hir_cost = G_s.nodes[nbr]['total_data']*u_data

            # find the min trust if the SIoT nbr joins the cluster c 
            nbr_min_t = get_min_trust(nbr, c['node'], c['p_level'])
            pri_cost = trust_to_cost(nbr_min_t)

            t_cost = cmp_cost + cmu_cost + hir_cost + pri_cost

            v_data = cmp_vaild_data(G_s.nodes[nbr]['d_data'])
            CP = v_data/t_cost
            if (CP >= CP_s):
                cnd_s = nbr
                CP_s = CP
                nbr_min_c = nbr_min
                min_trust = nbr_min_t

        cnd_SIoT = {'cluster_id': c['id'], 'node_id':cnd_s, 'CP_s': CP_s, "nbr_min_c":nbr_min_c,'min_trust': min_trust }
        cnd_SIoTs.append(cnd_SIoT)
    return cnd_SIoTs



#---------- find the highest CP SIoT from all condidate SIoTs --------------
def get_c_SIoT(SIoTs):
    SIoT = None
    CP = 0
    for s in SIoTs:
        if (s['CP_s'] > CP):
            SIoT = s
            CP = s['CP_s']
    return SIoT



#---------- check the data coverage constraint whether has satisfied  --------------
def check_data_coverage():
    for l in l_num_data:
        if(l!=0):
            return False
    return True


# --------- Step2. Data balancing for each cluster ~~~ Function ------------


#--------- find the cluster with max and min data size ------------
def find_max_min_c():
    max_c = 0
    max_d = sum([G_s.nodes[x]['total_data'] for x in clusters[0]['node']])
    min_c = 0
    min_d = sum([G_s.nodes[x]['total_data'] for x in clusters[0]['node']])
    
    for c in clusters:
        c_d = sum([G_s.nodes[x]['total_data'] for x in c['node']])
        
        if (c_d > max_d):
            max_d = c_d
            max_c = c['id']
            continue
        if (c_d < min_d):
            min_d = c_d
            min_c = c['id']
    
    return max_c, min_c, (max_d-min_d) < D 



#--------- find the leaf of the cluster ------------
def find_leaf_node(c):
    leaf = []
    for n in c['node']:
        if (c['graph'].degree(n) == 1):
            leaf.append(n)
    return leaf


#--------- check remove the node that has still satisfied data coverage constraint  ------------
def check_remove_node(l):
    lc = [total_num_data[i]-G_s.nodes[l]['d_data'][i] for i in range(0, label)]
#     print(l, lc)
    for i in range(0, label):
        if (total_num_data[i]-G_s.nodes[l]['d_data'][i] < B):
            return False
    return True
            


#--------- find the leaf node that satisfies data coverage constraint ------------
def find_cd_leaf_node(c):
    leaf = find_leaf_node(c)
    cd_leaf = []
    
    for l in leaf:
        if (check_remove_node(l)):
            cd_leaf.append(l)
            
    return cd_leaf



#--------- find the highest cost node in leaf ------------
def find_max_cost_leaf(leaf, c_id):
    cost = 0
    s_id = None
    for l in leaf:
        cmp = cmp_cost = G_s.nodes[l]['cmp']
        
        nbr = list(clusters[c_id]['graph'].neighbors(l))[0]
        cmu_cost = G_s[l][nbr]['cmu']
        hir_cost = G_s.nodes[l]['total_data']*u_data
        t_cost = cmp_cost + cmu_cost + hir_cost
#         print (l , t_cost)
        if (t_cost > cost):
            cost = t_cost
            s_id = l
    return s_id



#--------- find the leaf node that satisfies data coverage constraint ------------
def find_highCP_s(c):
    adj_c =set()
    for n in c['node']:
        adj_c = adj_c|set(G_s.neighbors(n))
    adj_c = adj_c - hired_SIoTs
    #     print(adj_c)
    cnd_s = None
    CP_s = 0
    nbr_min_c = None
    min_trust = 0
    for nbr in adj_c:       
        cmp_cost = G_s.nodes[nbr]['cmp']

        # find the SIoT that has min cmu edge connected the SIoT nbr in cluster c 
        nbr_min = get_min_cmu_edge(nbr,c['node'])
        cmu_cost = G_s[nbr][nbr_min]['cmu']
        hir_cost = G_s.nodes[nbr]['total_data']*u_data

        # find the min trust if the SIoT nbr joins the cluster c 
        nbr_min_t = get_min_trust(nbr, c['node'], c['p_level'])
        pri_cost = trust_to_cost(nbr_min_t)

        t_cost = cmp_cost + cmu_cost + hir_cost + pri_cost

        CP = G_s.nodes[nbr]['total_data']/t_cost
        if (CP >= CP_s):
            cnd_s = nbr
            CP_s = CP
            nbr_min_c = nbr_min
            min_trust = nbr_min_t

    cnd_SIoT = {'cluster_id': c['id'], 'node_id':cnd_s, 'CP_s': CP_s, "nbr_min_c":nbr_min_c,'min_trust': min_trust }
    
    return cnd_SIoT



# --------- Step 3. Cluster head(CH) decision and Rerouting ~~~functoin ------------

# --------- get the minimum privacy level of node n on path p and the node that let n has this privacy requirment ------------
def get_min_plevel(n, p):
    min_p = 1
    pri_n = None
    for i in p:
        if (get_2SIoT_trust(n, i) < min_p):
            min_p = get_2SIoT_trust(n, i)
            pri_n = i
    return min_p, pri_n


# --------- get the total cost if node n is cluster head ------------
def get_total_cost(n, c):
    nodes = c['node']
    s_ps = []    # record shortest paths of all node to node n
    total_cmu = 0
    total_pri = 0
    min_p_level = 1
    # get all shortest paths
    for i in nodes:
        if (i == n):
            continue
        s_ps.append(nx.shortest_path(c['graph'],source=i,target=n, weight='cmu'))
        
    # compute total communication cost
    for sp in s_ps:
        for i in range(1, len(sp)):
            total_cmu += G_s[sp[i-1]][sp[i]]['cmu']
    
    #find the min p level
    for sp in s_ps:
        min_p, _ = get_min_plevel(sp[0], sp)
        if (min_p < min_p_level):
            min_p_level = min_p
    
    #compute the total privacy cost
    total_pri = len(nodes)*trust_to_cost(min_p_level) 
#     print(min_p_level)

    return total_cmu + total_pri



# --------- get the node with minimum total cost in the cluster c ------------
def get_ch(c_ch, c):
    ch = None
    cost = 9999999
    for n in c_ch:
        n_cost = get_total_cost(n, c)
#         print (n_cost , n)
        if (n_cost<cost):
            cost = n_cost
            ch = n
    return ch



# --------- set node ch as the cluster head of the cluster c ------------
def set_ch(ch, c):
    c_id = c['id']
    nodes = c['node']
    s_ps = []    # record shortest paths of all node to node n
    total_cmp = sum([G_s.nodes[i]['cmp'] for i in nodes])
    total_cmu = 0
    total_hir = sum([G_s.nodes[i]['total_data'] for i in nodes])*u_data
    total_pri = 0
    min_p_level = 1
    # get all shortest paths
    for i in nodes:
        if (i == ch):
            continue
        s_ps.append(nx.shortest_path(c['graph'],source=i,target=ch, weight='cmu'))
        
    # compute total communication cost
    for sp in s_ps:
        for i in range(1, len(sp)):
            total_cmu += G_s[sp[i-1]][sp[i]]['cmu']
    
    #find the min p level
    for sp in s_ps:
        min_p, pri_n = get_min_plevel(sp[0], sp)
        
        #set node info
        c['graph'].nodes[sp[0]]['p_level'] = min_p
        c['graph'].nodes[sp[0]]['pri_n'] = pri_n
        
        if (min_p < min_p_level):
            min_p_level = min_p
    c['graph'].nodes[ch]['p_level'] = 1
    c['graph'].nodes[ch]['pri_n'] = ch
    
    
    #compute the total privacy cost
    total_pri = len(nodes)*trust_to_cost(min_p_level) 
    
    # set cluster info 
    clusters[c_id]['p_level'] = min_p_level
    clusters[c_id]['CH'] = ch
    clusters[c_id]['total_cmp'] = total_cmp
    clusters[c_id]['total_cmu'] = total_cmu
    clusters[c_id]['total_hir'] = total_hir
    clusters[c_id]['total_pri'] = total_pri
    clusters[c_id]['total_cst'] = total_cmp + total_cmu + total_hir + total_pri
    
    

# get the minimum privacy level node in the cluster c
def get_min_pl_n(c):
    
    min_n = None
    min_l = 1
    for i in list(c['node']):
#         print (i , c['graph'].nodes[i]['p_level'])
        if (c['graph'].nodes[i]['p_level'] < min_l):
            min_n = i
            min_l = c['graph'].nodes[i]['p_level']
    
    return min_n
    
# compute the total cost
def cmp_t_cost(G, ch):
    
    nodes = G.nodes()
    s_ps = []    # record shortest paths of all node to node n
    total_cmp = sum([G_s.nodes[i]['cmp'] for i in nodes])
    total_cmu = 0
    total_hir = sum([G_s.nodes[i]['total_data'] for i in nodes])*u_data
    total_pri = 0
    min_p_level = 1
    # get all shortest paths
    for i in nodes:
        if (i == ch):
            continue
        try:
            s_ps.append(nx.shortest_path(G,source=i,target=ch, weight='cmu'))
        except:
            return 100000
    # compute total communication cost
    for sp in s_ps:
        for i in range(1, len(sp)):
            total_cmu += G_s[sp[i-1]][sp[i]]['cmu']
    
    #find the min p level
    for sp in s_ps:
        min_p, pri_n = get_min_plevel(sp[0], sp)
        
        if (min_p < min_p_level):
            min_p_level = min_p
    
    
    #compute the total privacy cost
    total_pri = len(nodes)*trust_to_cost(min_p_level) 
    
    return total_cmp + total_cmu + total_pri + total_hir



#  rerouting
def rerouting(c):
    min_n = get_min_pl_n(c)
    pri_n = c['graph'].nodes[min_n]['pri_n']
    path = nx.shortest_path(c['graph'],source=min_n,target=pri_n, weight='cmu')
    
    G_c = nx.Graph(G_s)
    nodes = set(G_c.nodes) - set(c['node'])
    G_c.remove_nodes_from(list(nodes))
#     nx.draw(G_c)
    ori = None
    cnd = None
    t_cost = c['total_cst']
    
    for n in list(path)[:-1]:
        cnd_e = G_c.edges([n])
        ori_e = None
#         print(G_c.edges([n]))
        # find original edge of the node n 
        for e in cnd_e:
            if e in list(c['graph'].edges) or (e[1],e[0]) in list(c['graph'].edges):
                ori_e = e
        
        for e in cnd_e:
            # skip the original path 
            if e in list(c['graph'].edges) or (e[1],e[0]) in list(c['graph'].edges):
                continue
                
            G_d = nx.Graph(c['graph'])
#             print(e, ori_e)
            
            G_d.remove_edges_from([ori_e])
            
            G_d.add_edges_from([(e[0], e[1], G_s[e[0]][e[1]])])
            
            cost = cmp_t_cost(G_d, c['CH'])
            
            if (t_cost > cost):
                t_cost = cost
                ori = ori_e
                cnd = e
#             print(t_cost, c['total_cst'])
#     print(ori, cnd)
    return ori, cnd, ori != None




#  exchange the two edge 
def change_route(ori_e, rerout_e, c_id):
    clusters[c_id]['graph'].remove_edges_from([ori_e])        
    clusters[c_id]['graph'].add_edges_from([(rerout_e[0], rerout_e[1], G_s[rerout_e[0]][rerout_e[1]])])
    print ("change {} to {}".format(ori_e, rerout_e))
    
    
    
def update_c_info (c_id): 
    nodes = clusters[c_id]['node']
    ch = clusters[c_id]['CH']
    ori_cost = clusters[c_id]['total_cst']
    s_ps = []    # record shortest paths of all node to node n
    total_cmp = sum([G_s.nodes[i]['cmp'] for i in nodes])
    total_cmu = 0
    total_hir = sum([G_s.nodes[i]['total_data'] for i in nodes])*u_data
    total_pri = 0
    min_p_level = 1
    # get all shortest paths
    for i in nodes:
        if (i == ch):
            continue
        s_ps.append(nx.shortest_path(clusters[c_id]['graph'],source=i,target=ch, weight='cmu'))
        
    # compute total communication cost
    for sp in s_ps:
        for i in range(1, len(sp)):
            total_cmu += G_s[sp[i-1]][sp[i]]['cmu']
    
    #find the min p level
    for sp in s_ps:
        min_p, pri_n = get_min_plevel(sp[0], sp)
        
        #set node info
        clusters[c_id]['graph'].nodes[sp[0]]['p_level'] = min_p
        clusters[c_id]['graph'].nodes[sp[0]]['pri_n'] = pri_n
        
        if (min_p < min_p_level):
            min_p_level = min_p
    
    #compute the total privacy cost
    total_pri = len(nodes)*trust_to_cost(min_p_level) 
    
    # set cluster info 
    clusters[c_id]['p_level'] = min_p_level
    clusters[c_id]['total_cmp'] = total_cmp
    clusters[c_id]['total_cmu'] = total_cmu
    clusters[c_id]['total_hir'] = total_hir
    clusters[c_id]['total_pri'] = total_pri
    clusters[c_id]['total_cst'] = total_cmp + total_cmu + total_hir + total_pri
    
    return ori_cost - clusters[c_id]['total_cst']



def get_overall_cost():
    overall_cmp = sum([c['total_cmp'] for c in clusters])
    overall_cmu = sum([c['total_cmu'] for c in clusters])
    overall_hir = sum([c['total_hir'] for c in clusters])
    overall_pri = sum([c['total_pri'] for c in clusters])
    
    overall_cst = overall_cmp + overall_cmu + overall_hir + overall_pri
    
    return [overall_cmp, overall_cmu, overall_hir, overall_pri, overall_cst]