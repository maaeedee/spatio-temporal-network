import numpy as np 
import pandas as pd
import networkx as nx


# Calculate spatial distance for the spatiotemporal graph
def distances(gps_data, device_A, device_B, date, previous_date, process_type, win_size, dummy = 'False'):

    if dummy == 'True': 
        return distances_dummy(gps_data, device_A, device_B, date, previous_date, process_type, win_size)
    
    else:
        point_A_date = gps_data[(gps_data.Code == int(device_A)) & (gps_data.time >= (date - win_size)) & (gps_data.time <= (date + win_size))]
        point_B_date = gps_data[(gps_data.Code == int(device_B)) & (gps_data.time >= (date - win_size)) & (gps_data.time <= (date + win_size))]

        point_A_previous_date = gps_data[(gps_data.Code == int(device_A)) & (gps_data.time >= (previous_date - win_size)) & (gps_data.time <= (previous_date + win_size))]

        if process_type == 'temporal': 
            point1 = np.array((np.median(point_A_date.X), np.median(point_A_date.Y)))
            point2 = np.array((np.median(point_A_previous_date.X), np.median(point_A_previous_date.Y)))
        
        elif process_type == 'instant':
            point1 = np.array((np.median(point_A_date.X), np.median(point_A_date.Y)))
            point2 = np.array((np.median(point_B_date.X), np.median(point_B_date.Y)))

        distance = np.linalg.norm(point1 - point2)
        return distance

# Calculate dummy spatial distance for the spatiotemporal graph
def distances_dummy(gps_data, device_A, device_B, date, previous_date, process_type, win_size):
    point_A_date = gps_data[(gps_data.Code == (device_A)) & (gps_data.time == (date ))]
    point_B_date = gps_data[(gps_data.Code == (device_B)) & (gps_data.time == (date ))]

    point_A_previous_date = gps_data[(gps_data.Code == (device_A)) & (gps_data.time == (previous_date))]

    if process_type == 'temporal': 
        point1 = np.array(((point_A_date.X), (point_A_date.Y)))
        point2 = np.array(((point_A_previous_date.X), (point_A_previous_date.Y)))
    
    elif process_type == 'instant':
        point1 = np.array(((point_A_date.X), (point_A_date.Y)))
        point2 = np.array(((point_B_date.X), (point_B_date.Y)))

        if (len(point_A_date) == 0) | (len(point_B_date) == 0):
            distance = 0
            return distance

    distance = np.linalg.norm(point1 - point2)
    return distance


def attach_instance (device, date, last_seen, file, gps_data, analysis_type, process_type, win_size):
    previous_date=last_seen[device]
    if (previous_date != "")and (previous_date!= date):
        diff = abs(date - previous_date)

        if analysis_type == 'spatiotemporal':
            dist_devices = distances(gps_data, device, device, date, previous_date, process_type, win_size)
        else:
            dist_devices = 0
        file.write("%s_%d,%s_%d,%d,%f \r\n"%(device,previous_date,device,date,diff,dist_devices))

    last_seen[device]=date

# Find interactions in a raw edge data
def find_interactions(edge_file, edge_list_file, time_interval = 1):
    data = edge_file.copy()
    tagx = np.unique(data.tagX)
    tagy = np.unique(data.tagY)
    database = pd.DataFrame(columns = ['tagX', 'tagY', 'StartTime', 'Weight'])
    cntr = 0 
    for y in tagy:
        for x in tagx:
            temp = data[(data.tagX==x)&(data.tagY==y)].copy()
            temp = temp.reset_index(drop=True)

            if len(temp)>0:
                try: 
                    indx = list(np.squeeze(np.where(np.diff(temp.time)>time_interval)))
                except: 
                    indx = np.squeeze(np.where(np.diff(temp.time)>time_interval))
                    # print('0: ', indx)
                    if np.shape(indx)==0:
                        indx = [0,len(temp)]
                        # print('1: ', indx)
                    else:
                        indx = [np.squeeze(np.where(np.diff(temp.time)>1)).tolist()]
                        # print('2: ', indx)

                # final index
                indx.append(len(temp))
                # first index 
                indx.insert(0,0)
                
                for i in range(0,len(indx)-1): 

                    if i==0:
                        temp1 = temp.loc[indx[i]:indx[i+1],:]
                        temp1 = temp1.reset_index(drop=True)
                        if len (temp1)>0:
                            database.loc[cntr, 'tagX'] = temp1['tagX'][0]
                            database.loc[cntr, 'tagY'] = temp1['tagY'][0]
                            database.loc[cntr, 'StartTime'] = (temp1['time'][0]).astype(int)
                            database.loc[cntr, 'Weight'] = len(temp1['time'])
                            cntr = cntr+1
                        
                    else:
                        temp1 = temp.loc[indx[i]+1:indx[i+1],:]
                        temp1 = temp1.reset_index(drop=True)
                        if len (temp1)>0:
                            database.loc[cntr, 'tagX'] = temp1['tagX'][0]
                            database.loc[cntr, 'tagY'] = temp1['tagY'][0]
                            database.loc[cntr, 'StartTime'] = (temp1['time'][0]).astype(int)
                            database.loc[cntr, 'Weight'] = len(temp1['time'])
                            cntr = cntr+1

    file_name = 'interactions_' + edge_list_file
    database['StartTime'] = database['StartTime'].astype(int)
    database['tagX'] = database['tagX'].astype(str)
    database['tagY'] = database['tagY'].astype(str)
    database['Weight'] = database['Weight'].astype(int)
    database.to_csv('./Data/' + file_name, index = False)
    return database, file_name

# Create spatiotemporal edges
def preprocess(data, input_file, gps_data, analysis_type, seperator, win_size):
    temporal_graph_file = open("tempdata.txt","w+")

    column_values = data[["tagX", "tagY"]].values.ravel()
    unique_values =  pd.unique(column_values.astype(str))
    last_seen = dict.fromkeys((unique_values),'')

    try:
        next(input_file)
        for line in input_file:
            
            line=line.replace('"', '').strip()
            items=line.split(seperator)

            sender = items[0]
            recepient = items[1].replace("\n","")
            date = float(items[2])
    

            process_type = 'temporal'
            attach_instance(sender, date, last_seen, temporal_graph_file, gps_data, analysis_type, process_type, win_size)
            attach_instance(recepient, date, last_seen, temporal_graph_file, gps_data, analysis_type, process_type, win_size)

            if analysis_type == 'spatiotemporal':
                process_type = 'instant'
                dist_devices = distances(gps_data, sender, recepient, date, date, process_type, win_size)
            else: 
                dist_devices = 0
            temporal_graph_file.write("%s_%d,%s_%d,0,%f \r\n"%(sender,date,recepient, date, dist_devices))
            temporal_graph_file.write("%s_%d,%s_%d,0,%f \r\n"%(recepient,date,sender, date, dist_devices))

    finally:
        input_file.close()
        temporal_graph_file.close()

    return

# Read temporal edges in the form of a graph
def read_as_graph(sender, receiver, weights, analysis_type, pg):

    temp_graph = pd.read_csv('tempdata.txt', sep=",", header=None)
    temp_graph.columns = [sender, receiver, 'time', 'distance']
    
    # Quick back and forth between string and float to exclude all the 'nan ' values from distance
    temp_graph.distance = temp_graph.distance.astype('string')
    temp_graph = temp_graph[temp_graph.distance != 'nan ']
    temp_graph.distance = temp_graph.distance.astype(float)
    Graph = nx.from_pandas_edgelist(temp_graph, source = sender, target = receiver, edge_attr = weights, create_using=nx.DiGraph())
    return Graph


# Calculate costs for the spatiotemporal shortest path algorithm
def cost_finder(all_path_x, NODEy):
    null = 0
    all_costs = 0
    v_counter = 0 
    for i in all_path_x: 
        dicts = all_path_x[i]
        try: 
            cost = np.min([dicts[i] for i in dicts.keys() if i.startswith(NODEy)])
            all_costs = cost+ all_costs 
            v_counter = v_counter +1
        except:
            null = null +1 
    return all_costs, v_counter, null

# Calculate the spatiotemporal variables via shortest path algorithm
def calculate_tables(Grph, weights):

    # Find all unweighted path
    all_path = dict(nx.all_pairs_dijkstra(Grph))

    # Find all weighted path
    all_path_weighted = dict(nx.all_pairs_dijkstra(Grph, weight = weights))

    # store all temporal nodes 
    temp_nodes = list(all_path.keys())

    # Store all the original unique nodes
    unix_nodes = np.unique([i.split('_')[0] for i in temp_nodes])

    # Start creating spatiotemporal matrix
    V_mat = pd.DataFrame(columns=unix_nodes, index = unix_nodes)
    G_mat = pd.DataFrame(columns=unix_nodes, index = unix_nodes)
    P_mat = pd.DataFrame(columns=unix_nodes, index = unix_nodes)

    # Calculating the shortest path algorithm
    for i, NODEx in enumerate(unix_nodes): 
        for j, NODEy in enumerate(unix_nodes): 

            # G and V calculation
            # Find all path that involves node X
            all_path_x = {k:v[0] for k,v in all_path.items() if k.split('_')[0]==NODEx}
            temp_node_x = all_path_x.keys()
            all_costs, v_counter, null = cost_finder(all_path_x, NODEy)

            # P calculation
            # Find all path that involves node X
            P_all_path_x = {k:v[0] for k,v in all_path_weighted.items() if k.split('_')[0]==NODEx}
            temp_node_x = P_all_path_x.keys()
            P_all_costs, P_v_counter, P_null = cost_finder(P_all_path_x, NODEy)

            # Find all the available temporal nodes
            n = len(temp_node_x) 

            try: 
                G = all_costs/(n- null)
                V = v_counter/n
                P = P_all_costs/(n-P_null)
            except: 
                G = np.nan
                V = 0 
                P = np.nan

            G_mat.loc[NODEx,NODEy] = G
            G_inout = G_mat.copy()
            V_mat.loc[NODEx,NODEy] = V
            V_inout = V_mat.copy()
            P_mat.loc[NODEx,NODEy] = P
            P_inout = P_mat.copy()


    # Calculate in and out variables
    P_inout['Pout']=(P_mat.sum(1))/(P_mat.count(axis=1)-1)
    P_inout.loc['Pin']=P_mat.sum()/(P_mat.count(axis=0)-1)
    V_inout['Vout']=(V_mat.sum(1))/(V_mat.count(axis=1))
    V_inout.loc['Vin']=V_mat.sum()/(V_mat.count(axis=0))
    G_inout['Gout']=(G_mat.sum(1))/(G_mat.count(axis=1)-1)
    G_inout.loc['Gin']=G_mat.sum()/(G_mat.count(axis=0)-1)

    return P_inout, P_mat, V_inout, V_mat, G_inout, G_mat
