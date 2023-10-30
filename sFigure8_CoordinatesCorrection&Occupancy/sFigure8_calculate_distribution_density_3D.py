# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:38:11 2023

@author: Jialin Ye
@institution: SIAT
@Contact_email: jl.ye@siat.ac.cn

"""

import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
import scipy.stats as st
import matplotlib.cm as cm
import matplotlib.colors as mcolors


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\sFigure8_CoordinatesCorrection&Occupancy'

skip_file_list = [1,3,28,29,110,122] 

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Feature_space_path = get_path(InputData_path_dir,'Feature_Space.csv')


animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


dataset = Night_lightOff_info
dataset_name = 'Night_lightOff'

output_dir1 = output_dir + '/{}/distribution_density_normalized_all60min'.format(dataset_name)
output_dir2 = output_dir + '/{}/distribution_density_normalized_Every10min'.format(dataset_name) 
if not os.path.exists(output_dir1):                                                                       
    os.mkdir(output_dir1)
if not os.path.exists(output_dir2):                                                                       
    os.mkdir(output_dir2)

coor_data_list = []
for index in list(dataset.index):
    video_index = dataset.loc[index,'video_index']
    gender = dataset.loc[index,'gender']
    ExperimentTime = dataset.loc[index,'ExperimentTime']

    coor_data = pd.read_csv(Movement_Label_path[video_index],index_col=0)
    coor_data['time'] = coor_data.index/(30*60) + 1
    coor_data['time'] = coor_data['time'].apply(lambda x: int(x))
    coor_data_list.append(coor_data)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

coor_data_all = pd.concat(coor_data_list,axis=0)
coor_data_all.reset_index(drop=True,inplace=True)


def normalized_all60min(coor_data_all):
    x = coor_data_all['back_x']
    y = coor_data_all['back_y']
    xmin, xmax = -10, 511
    ymin, ymax = -10, 511
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    
    max_value = np.max(f)
    min_value = np.min(f)
    
    f = (f - min_value)/(max_value-min_value)
    
    cmap1 = cm.RdBu_r
    norm1 = mcolors.Normalize(vmin = f.min(), vmax= f.max())
    
    print(f.min(),f.max())
    
    
    for i in range(10,61,10):
        print(i)
        
        temp_df = coor_data_all[coor_data_all['time']<=i]
        x = temp_df['back_x']
        y = temp_df['back_y']
        
        xmin, xmax = -10, 511
        ymin, ymax = -10, 511
        
        
        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        f = (f - min_value)/(max_value-min_value)
        print(f.min(),f.max())
        colors =  cmap1(norm1(f))
        
        fig = plt.figure(figsize=(10,10),dpi=600)
        
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(xx,yy,f,facecolors=colors)
        ax.set_zlim(0,1.2)
        ax.view_init(60, 35)
        ax.axis('off')
        plt.savefig(r'{}\{}_{}min.png'.format(output_dir1,dataset_name,i),dpi=300,transparent=True)
        plt.show()


def normalized_Every10min(coor_data_all):
    end = 0
    start = 0
    for i in range(10,61,10):
         end = i
         
         temp_df = coor_data_all[ (coor_data_all['time']>start) & (coor_data_all['time']<=end) ]
         x = temp_df['back_x']
         y = temp_df['back_y']
         
         xmin, xmax = -10, 510
         ymin, ymax = -10, 510
         
         
         # Peform the kernel density estimate
         xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
         positions = np.vstack([xx.ravel(), yy.ravel()])
         values = np.vstack([x, y])
         kernel = st.gaussian_kde(values)
         f = np.reshape(kernel(positions).T, xx.shape)
         max_value = np.max(f)
         min_value = np.min(f)
         f = (f - min_value)/(max_value-min_value)
         print(f.min(),f.max())
         cmap1 = cm.RdBu_r
         norm1 = mcolors.Normalize(vmin = f.min(), vmax= f.max())
         colors =  cmap1(norm1(f))
        
         fig = plt.figure(figsize=(10,10),dpi=600)
         
         ax = fig.add_subplot(111,projection='3d')
         ax.plot_surface(xx,yy,f,facecolors=colors)
         ax.set_zlim(0,1.2)
         ax.view_init(60, 35)
         ax.axis('off')
         plt.savefig(r'{}\{}_{}min.png'.format(output_dir1,dataset_name,i),dpi=300,transparent=True)
         start = end
         plt.show()


normalized_all60min(coor_data_all)
normalized_Every10min(coor_data_all)
