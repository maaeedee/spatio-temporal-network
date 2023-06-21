import pandas as pd
import numpy as np 
import networkx as nx
from spatiotemporal_analysis.generate_graph import preprocess, read_as_graph, calculate_tables
import json
import os 


if __name__ == '__main__':

    edge_list_file = 'dummy_example.csv'
    gps_file = 'dummy_example_gps_data.csv'
    prefix = 'dummy_'
    win_size = 10

    analysis_type = 'spatiotemporal'

    # set path
    root = os.getcwd() + '/'
    result_path = root + '/Results/'

    # Open the Json file - including meta data for the analysis
    json_file = open('metadata.json')
    meta_data = json.load(json_file)

    # initialise variables with meta data information
    meta_data = meta_data[edge_list_file]
    dummy = meta_data['dummy']

    # Read interactions data
    file_name = edge_list_file
    database = pd.read_csv(root + '/Data/' + file_name)
    input_file = open('./Data/' + file_name, "r")


    # Generate temporal Graph 
    seperator = ','
    gps_data = pd.read_csv(root + 'Data/' + gps_file)

    # Preprocess data to create spatiotemporal graph
    preprocess(database, input_file,  gps_data, analysis_type, seperator, win_size, dummy)

    # Calculate Average Spatiotemporal and Geodesic Proximity
    weights = 'distance'
    sender = "sender"
    receiver = "receiver"
    analysis = 'Cartesian_analysis'

    # Read temporal data as a graph
    Grph = read_as_graph(sender, receiver, weights, analysis, prefix)

    # Creating the required data
    P_inout, P_mat, V_inout, V_mat, G_inout, G_mat = calculate_tables(Grph, weights)
    P_inout.to_csv(result_path + prefix +  'Average_Spatio-temporal_metric_inout.csv')
    P_mat.to_csv(result_path + prefix +  'Average_Spatio-temporal_metric_matrix.csv')

    V_inout.to_csv(result_path + prefix +  'Average_Availability_metric_inout.csv')
    V_mat.to_csv(result_path + prefix +  'Average_Availability_metric_matrix.csv')

    G_inout.to_csv(result_path + prefix +  'Average_Geodesic_metric_inout.csv')
    G_mat.to_csv(result_path + prefix +  'Average_Geodesic_metric_matrix.csv')


    # Calculate Average Temporal Proximity
    weights = 'time'

    # Read temporal data as a graph
    Grph = read_as_graph(sender, receiver, weights, analysis, prefix)

    # Creating the required data
    P_inout, P_mat, V_inout, V_mat, G_inout_1, G_mat = calculate_tables(Grph, weights)
    P_inout.to_csv(result_path + prefix +  'Average_Temporal_metric_inout.csv')
    P_mat.to_csv(result_path + prefix +  'Average_Temporal_metric_matrix.csv')

