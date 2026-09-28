import math
import scipy as sc
import time
import os
from scipy.stats import norm
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, bar_format="{bar} {percentage:3.0f}%", ncols=50)
import seaborn as sns
import statistics
from pycop import empirical
import statsmodels.formula.api as smf
import os
from pymnet import *
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Return.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df_filtered = df.set_index('date').dropna()

adj_matrices_corr = []
adj_matrices_corr_NW = []
adj_matrices_ken = []
adj_matrices_ken_NW = []
adj_matrices_par = []
adj_matrices_par_NW = []
adj_matrices_tail = []
adj_matrices_tail_NW=[]
data_rolling = df_filtered

correlation_matrix = data_rolling.corr(method='pearson')

kendall_corr_matrix = data_rolling.corr(method='kendall')

inv_corr_matrix = np.linalg.inv(correlation_matrix)
partial_corr_matrix = pd.DataFrame(-inv_corr_matrix / np.sqrt(np.outer(np.diag(inv_corr_matrix), np.diag(inv_corr_matrix))))
partial_corr_matrix.columns = correlation_matrix.columns
partial_corr_matrix.index = correlation_matrix.index

tail_corr_matrix = pd.DataFrame(np.nan, index=data_rolling.columns, columns=data_rolling.columns)
for i in range(len(data_rolling.columns)):
    for j in range(i, len(data_rolling.columns)):
        stock_i = data_rolling.columns[i]
        stock_j = data_rolling.columns[j]

        tail_corr = empirical(data_rolling[[stock_i, stock_j]].T.values).LTDC(0.1)

        tail_corr_matrix.loc[stock_i, stock_j] = tail_corr
        tail_corr_matrix.loc[stock_j, stock_i] = tail_corr
np.fill_diagonal(tail_corr_matrix.values, 1)

G_corr = nx.Graph()
for i, row in correlation_matrix.iterrows():
    for j, value in row.items():
        if i != j:
            distance = np.sqrt(2 * (1 - (abs(value))))
            G_corr.add_edge(i, j, weight = distance)

mst = nx.minimum_spanning_tree(G_corr, weight='weight')
MST_corr = nx.Graph()

for u, v, data in mst.edges(data=True):
    distance = data['weight']
    value = 1 - (distance ** 2) / 2
    MST_corr.add_edge(u, v, weight=value)
    adj_matrix_corr = nx.to_numpy_array(MST_corr)
    adj_matrix_corr_NW = (adj_matrix_corr > 0).astype(int)
pear_df=pd.DataFrame(adj_matrix_corr_NW)
pear_df.columns=df_filtered.columns
pear_df.index=df_filtered.columns

G_ken = nx.Graph()
for i, row in kendall_corr_matrix.iterrows():
    for j, value in row.items():
        if i != j:
            distance = np.sqrt(2 * (1 - (abs(value))))
            G_ken.add_edge(i, j, weight = distance)

mst = nx.minimum_spanning_tree(G_ken, weight='weight')
MST_ken = nx.Graph()

for u, v, data in mst.edges(data=True):
    distance = data['weight']
    value = 1 - (distance ** 2) / 2
    MST_ken.add_edge(u, v, weight=value)
    adj_matrix_ken = nx.to_numpy_array(MST_ken)
    adj_matrix_ken_NW = (adj_matrix_ken > 0).astype(int)
ken_df=pd.DataFrame(adj_matrix_ken_NW)
ken_df.columns=df_filtered.columns
ken_df.index=df_filtered.columns

G_par = nx.Graph()
for i, row in partial_corr_matrix.iterrows():
    for j, value in row.items():
        if i != j:
            distance = np.sqrt(2 * (1 - (abs(value))))
            G_par.add_edge(i, j, weight = distance)

mst = nx.minimum_spanning_tree(G_par, weight='weight')
MST_par = nx.Graph()

for u, v, data in mst.edges(data=True):
    distance = data['weight']
    value = 1 - (distance ** 2) / 2
    MST_par.add_edge(u, v, weight=value)
    adj_matrix_par = nx.to_numpy_array(MST_par)
    adj_matrix_par_NW = (adj_matrix_par > 0).astype(int)
par_df=pd.DataFrame(adj_matrix_par_NW)
par_df.columns=df_filtered.columns
par_df.index=df_filtered.columns

G_tail = nx.Graph()
for i, row in tail_corr_matrix.iterrows():
    for j, value in row.items():
        if i != j:
            distance = np.sqrt(2 * (1 - abs((value))))
            G_tail.add_edge(i, j, weight = distance)

mst = nx.minimum_spanning_tree(G_tail, weight='weight')
MST_tail = nx.Graph()

for u, v, data in mst.edges(data=True):
    distance = data['weight']
    value = 1 - (distance ** 2) / 2
    MST_tail.add_edge(u, v, weight=value)
    adj_matrix_tail = nx.to_numpy_array(MST_tail)
    adj_matrix_tail_NW = (adj_matrix_tail > 0).astype(int)
tail_df=pd.DataFrame(adj_matrix_tail_NW)
tail_df.columns=df_filtered.columns
tail_df.index=df_filtered.columns

four_layer_net=MultilayerNetwork(aspects=1)
four_layer_net.add_node('AEX')
four_layer_net.add_node('AORD')
four_layer_net.add_node('BFX')
four_layer_net.add_node('BSESN')
four_layer_net.add_node('BVSP')
four_layer_net.add_node('FCHI')
four_layer_net.add_node('FTSE')
four_layer_net.add_node('GDAXI')
four_layer_net.add_node('GSPTSE')
four_layer_net.add_node('IXIC')
four_layer_net.add_node('KS11')
four_layer_net.add_node('KSE')
four_layer_net.add_node('OMXC20')
four_layer_net.add_node('OMXHPI')
four_layer_net.add_node('OMXSPI')
four_layer_net.add_node('OSEAX')
four_layer_net.add_node('RUT')
four_layer_net.add_node('SSMI')
four_layer_net.add_layer("Pearson layer")
four_layer_net['AEX','AORD','Pearson layer','Pearson layer']=1
four_layer_net['AEX','BFX','Pearson layer','Pearson layer']=1
four_layer_net['AORD','BVSP','Pearson layer','Pearson layer']=1
four_layer_net['AORD','KSE','Pearson layer','Pearson layer']=1
four_layer_net['AORD','OMXC20','Pearson layer','Pearson layer']=1
four_layer_net['BSESN','BVSP','Pearson layer','Pearson layer']=1
four_layer_net['BSESN','FCHI','Pearson layer','Pearson layer']=1
four_layer_net['BSESN','FTSE','Pearson layer','Pearson layer']=1
four_layer_net['BVSP','GSPTSE','Pearson layer','Pearson layer']=1
four_layer_net['BVSP','IXIC','Pearson layer','Pearson layer']=1
four_layer_net['BVSP','OMXHPI','Pearson layer','Pearson layer']=1
four_layer_net['GDAXI','GSPTSE','Pearson layer','Pearson layer']=1
four_layer_net['IXIC','KS11','Pearson layer','Pearson layer']=1
four_layer_net['KS11','OMXSPI','Pearson layer','Pearson layer']=1
four_layer_net['OMXHPI','RUT','Pearson layer','Pearson layer']=1
four_layer_net['OMXHPI','SSMI','Pearson layer','Pearson layer']=1
four_layer_net['OMXSPI','OSEAX','Pearson layer','Pearson layer']=1

four_layer_net.add_layer("Kendall layer")
four_layer_net['AEX','AORD','Kendall layer','Kendall layer']=1
four_layer_net['AORD','FTSE','Kendall layer','Kendall layer']=1
four_layer_net['AORD','GSPTSE','Kendall layer','Kendall layer']=1
four_layer_net['AORD','KSE','Kendall layer','Kendall layer']=1
four_layer_net['AORD','OMXC20','Kendall layer','Kendall layer']=1
four_layer_net['BFX','BSESN','Kendall layer','Kendall layer']=1
four_layer_net['BFX','BVSP','Kendall layer','Kendall layer']=1
four_layer_net['BFX','FCHI','Kendall layer','Kendall layer']=1
four_layer_net['BSESN','GSPTSE','Kendall layer','Kendall layer']=1
four_layer_net['BSESN','RUT','Kendall layer','Kendall layer']=1
four_layer_net['GDAXI','GSPTSE','Kendall layer','Kendall layer']=1
four_layer_net['IXIC','KS11','Kendall layer','Kendall layer']=1
four_layer_net['KS11','OSEAX','Kendall layer','Kendall layer']=1
four_layer_net['KSE','OMXHPI','Kendall layer','Kendall layer']=1
four_layer_net['OMXHPI','OMXSPI','Kendall layer','Kendall layer']=1
four_layer_net['OMXHPI','SSMI','Kendall layer','Kendall layer']=1
four_layer_net['OMXSPI','OSEAX','Kendall layer','Kendall layer']=1

four_layer_net.add_layer("Partial layer")
four_layer_net['AEX','AORD','Partial layer','Partial layer']=1
four_layer_net['AEX','BFX','Partial layer','Partial layer']=1
four_layer_net['AORD','KS11','Partial layer','Partial layer']=1
four_layer_net['AORD','KSE','Partial layer','Partial layer']=1
four_layer_net['BSESN','BVSP','Partial layer','Partial layer']=1
four_layer_net['BSESN','FCHI','Partial layer','Partial layer']=1
four_layer_net['BSESN','FTSE','Partial layer','Partial layer']=1
four_layer_net['BVSP','IXIC','Partial layer','Partial layer']=1
four_layer_net['BVSP','OMXC20','Partial layer','Partial layer']=1
four_layer_net['FCHI','GDAXI','Partial layer','Partial layer']=1
four_layer_net['GSPTSE','IXIC','Partial layer','Partial layer']=1
four_layer_net['IXIC','KSE','Partial layer','Partial layer']=1
four_layer_net['IXIC','OMXHPI','Partial layer','Partial layer']=1
four_layer_net['OMXC20','OSEAX','Partial layer','Partial layer']=1
four_layer_net['OMXHPI','OMXSPI','Partial layer','Partial layer']=1
four_layer_net['OSEAX','RUT','Partial layer','Partial layer']=1
four_layer_net['RUT','SSMI','Partial layer','Partial layer']=1

four_layer_net.add_layer("Tail layer")
four_layer_net['AEX','AORD','Tail layer','Tail layer']=1
four_layer_net['AORD','FTSE','Tail layer','Tail layer']=1
four_layer_net['AORD','KS11','Tail layer','Tail layer']=1
four_layer_net['AORD','KSE','Tail layer','Tail layer']=1
four_layer_net['AORD','OMXC20','Tail layer','Tail layer']=1
four_layer_net['BFX','BSESN','Tail layer','Tail layer']=1
four_layer_net['BFX','BVSP','Tail layer','Tail layer']=1
four_layer_net['BFX','FCHI','Tail layer','Tail layer']=1
four_layer_net['BSESN','OMXC20','Tail layer','Tail layer']=1
four_layer_net['BSESN','OMXHPI','Tail layer','Tail layer']=1
four_layer_net['BSESN','RUT','Tail layer','Tail layer']=1
four_layer_net['FTSE','GDAXI','Tail layer','Tail layer']=1
four_layer_net['GSPTSE','OMXSPI','Tail layer','Tail layer']=1
four_layer_net['IXIC','OMXSPI','Tail layer','Tail layer']=1
four_layer_net['OMXHPI','OMXSPI','Tail layer','Tail layer']=1
four_layer_net['OSEAX','RUT','Tail layer','Tail layer']=1
four_layer_net['RUT','SSMI','Tail layer','Tail layer']=1

four_layer_net['AEX','AEX','Pearson layer','Kendall layer']=1
four_layer_net['AORD','AORD','Pearson layer','Kendall layer']=1
four_layer_net['BFX','BFX','Pearson layer','Kendall layer']=1
four_layer_net['BSESN','BSESN','Pearson layer','Kendall layer']=1
four_layer_net['BVSP','BVSP','Pearson layer','Kendall layer']=1
four_layer_net['FCHI','FCHI','Pearson layer','Kendall layer']=1
four_layer_net['FTSE','FTSE','Pearson layer','Kendall layer']=1
four_layer_net['GDAXI','GDAXI','Pearson layer','Kendall layer']=1
four_layer_net['GSPTSE','GSPTSE','Pearson layer','Kendall layer']=1
four_layer_net['IXIC','IXIC','Pearson layer','Kendall layer']=1
four_layer_net['KS11','KS11','Pearson layer','Kendall layer']=1
four_layer_net['KSE','KSE','Pearson layer','Kendall layer']=1
four_layer_net['OMXC20','OMXC20','Pearson layer','Kendall layer']=1
four_layer_net['OMXHPI','OMXHPI','Pearson layer','Kendall layer']=1
four_layer_net['OMXSPI','OMXSPI','Pearson layer','Kendall layer']=1
four_layer_net['OSEAX','OSEAX','Pearson layer','Kendall layer']=1
four_layer_net['RUT','RUT','Pearson layer','Kendall layer']=1
four_layer_net['SSMI','SSMI','Pearson layer','Kendall layer']=1

four_layer_net['AEX','AEX','Kendall layer','Partial layer']=1
four_layer_net['AORD','AORD','Kendall layer','Partial layer']=1
four_layer_net['BFX','BFX','Kendall layer','Partial layer']=1
four_layer_net['BSESN','BSESN','Kendall layer','Partial layer']=1
four_layer_net['BVSP','BVSP','Kendall layer','Partial layer']=1
four_layer_net['FCHI','FCHI','Kendall layer','Partial layer']=1
four_layer_net['FTSE','FTSE','Kendall layer','Partial layer']=1
four_layer_net['GDAXI','GDAXI','Kendall layer','Partial layer']=1
four_layer_net['GSPTSE','GSPTSE','Kendall layer','Partial layer']=1
four_layer_net['IXIC','IXIC','Kendall layer','Partial layer']=1
four_layer_net['KS11','KS11','Kendall layer','Partial layer']=1
four_layer_net['KSE','KSE','Kendall layer','Partial layer']=1
four_layer_net['OMXC20','OMXC20','Kendall layer','Partial layer']=1
four_layer_net['OMXHPI','OMXHPI','Kendall layer','Partial layer']=1
four_layer_net['OMXSPI','OMXSPI','Kendall layer','Partial layer']=1
four_layer_net['OSEAX','OSEAX','Kendall layer','Partial layer']=1
four_layer_net['RUT','RUT','Kendall layer','Partial layer']=1
four_layer_net['SSMI','SSMI','Kendall layer','Partial layer']=1

four_layer_net['AEX','AEX','Partial layer','Tail layer']=1
four_layer_net['AORD','AORD','Partial layer','Tail layer']=1
four_layer_net['BFX','BFX','Partial layer','Tail layer']=1
four_layer_net['BSESN','BSESN','Partial layer','Tail layer']=1
four_layer_net['BVSP','BVSP','Partial layer','Tail layer']=1
four_layer_net['FCHI','FCHI','Partial layer','Tail layer']=1
four_layer_net['FTSE','FTSE','Partial layer','Tail layer']=1
four_layer_net['GDAXI','GDAXI','Partial layer','Tail layer']=1
four_layer_net['GSPTSE','GSPTSE','Partial layer','Tail layer']=1
four_layer_net['IXIC','IXIC','Partial layer','Tail layer']=1
four_layer_net['KS11','KS11','Partial layer','Tail layer']=1
four_layer_net['KSE','KSE','Partial layer','Tail layer']=1
four_layer_net['OMXC20','OMXC20','Partial layer','Tail layer']=1
four_layer_net['OMXHPI','OMXHPI','Partial layer','Tail layer']=1
four_layer_net['OMXSPI','OMXSPI','Partial layer','Tail layer']=1
four_layer_net['OSEAX','OSEAX','Partial layer','Tail layer']=1
four_layer_net['RUT','RUT','Partial layer','Tail layer']=1
four_layer_net['SSMI','SSMI','Partial layer','Tail layer']=1

fig=draw(four_layer_net,
         layout="spring",
         layergap=0.87,
         camera_dist=10,
         nodeSizeRule={"rule":"degree","propscale":0.05},
         figsize=(14,19),
         layerColorDict={
             "Pearson layer": "lightgrey",
             "Kendall layer": "lavender",
             "Partial layer": "lightgreen",
             "Tail layer": "lightsalmon"
                          },
         nodeColorRule={"rule":"degree","colormap":"hsv","scaleby":0.17},
         edgeColorRule={'rule': 'edgetype', 'inter': 'gray', 'intra': 'darkorange'},
         azim=-30,
         elev=23,
         layerPadding=0.5,
         defaultEdgeWidth=2,
         defaultLayerAlpha=0.55,
         defaultLayerLabelSize=17,
         defaultNodeLabelSize=10,
         layerOrderDict = {"Pearson layer": 1, "Kendall layer": 2, "Partial layer": 3, "Tail layer": 4},
         defaultLayerLabelLoc=(0.7,1),
         edgeStyleRule={'inter': ':', 'intra': '-', 'rule': 'edgetype'}
         )
plt.savefig(OUTPUT_DIR / "Fig_02_multilayer_networks.pdf", dpi=900, bbox_inches="tight")
