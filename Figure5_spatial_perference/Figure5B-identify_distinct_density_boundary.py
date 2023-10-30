# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:06:07 2023

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
from scipy.signal import find_peaks,peak_widths,peak_prominences
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from sklearn import preprocessing
from scipy.interpolate import make_interp_spline



InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data" 
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure5_spatial_perference\Occupancy_in_different_areas'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)

coordinates_file_path = get_path(InputData_path_dir,'normalized_coordinates_back_XY.csv')


skip_file_list = [1,3,28,29,110,122] 
animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


def calculate_slope(x1,y1,x2,y2):
    x = (y2 - y1) / (x2 - x1)
    return abs(x)
def get_slope(x,y):
    slopee_list = []
    for i in range(1,len(x)):
        x1 = x[i-1]
        x2 = x[i]
        y1 = y[i-1]
        y2 = y[i]
        slope = calculate_slope(x1,y1,x2,y2)
        slopee_list.append(slope)
    slopee_list.insert(0, slopee_list[0])
    return(slopee_list)



dataset = animal_info


coor_data_list = []
for index in dataset.index:
    video_index = dataset.loc[index,'video_index']
    gender = dataset.loc[index,'gender']
    ExperimentTime = dataset.loc[index,'ExperimentTime']
    coor_data = pd.read_csv(coordinates_file_path[video_index],index_col=0)
    coor_data_list.append(coor_data)


def find_cross_point(height,curve,x_smooth):   
    start_point = 0
    end_point = 0
    for i in range(1,len(curve)):
        x1 = curve[i-1]
        x2 = curve[i]
        
        if (x1 <= height) & (height <= x2):
            start_point = i
        elif (x2 <= height) & (height <= x1):
            end_point = i
    return(start_point,end_point)


coor_data_all = pd.concat(coor_data_list,axis=0)
coor_data_all.reset_index(drop=True,inplace=True)

origin_x = origin_y = 250
coor_data_all['back_x'] = coor_data_all['back_x'] - origin_x
coor_data_all['back_y'] = coor_data_all['back_y'] - origin_y


X_range =  np.arange(1,250)

y_list = []
y_normal_list = []
r2_list = []
for i in X_range:
    S = np.square(i)
    y = len(coor_data_all[(abs(coor_data_all['back_x'])<=i) & (abs(coor_data_all['back_y'])<=i)])
    y_normal = float(y)/S
    y_list.append(y)
    y_normal_list.append(y_normal)
    r2_list.append(S)
    
y_list = np.array(y_list)
y_normal_list = np.array(y_normal_list)

#slope1 = np.array(get_slope(X_range,y_list)) 
slope1 = get_slope(X_range,y_list)   
max_id1 = slope1.index(max(slope1))
peaks1, _ = find_peaks(slope1,height=100000,prominence=10000)                   ## all data, 10000, 10000
results_half = peak_widths(slope1, peaks1, rel_height=0.9)
prominences = peak_prominences(slope1, peaks1)[0]
prominences_high = max(slope1) -prominences[0]*0.9

x_smooth = np.linspace(X_range.min(), X_range.max(), 10000)                    
y_smooth = make_interp_spline(X_range, slope1)(x_smooth)

start_point, end_point = find_cross_point(prominences_high,y_smooth,x_smooth)

#slope2 = np.array(get_slope(X_range,y_normal_list))
slope2 = get_slope(X_range,y_normal_list)
max_id2 = slope2.index(max(slope2))
peaks2, _ = find_peaks(slope2,height=1,prominence=0.5)

#slope3 = np.array(get_slope(r2_list,y_list))
slope3 = get_slope(r2_list,y_list)
max_id3 = slope3.index(max(slope3))
peaks3, _ = find_peaks(slope3,height=200,prominence=10)

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,2),dpi=300,constrained_layout=True,sharex=False)
sns.kdeplot(coor_data_all['back_x'],color='#1E88E5',lw = 5)
sns.kdeplot(coor_data_all['back_x'],color='#1E88E5',lw = 0,fill = True)
plt.title('back_x distribution')
plt.axis('off')
plt.savefig(r'{}/ALL_back_x_distribution_density.png'.format(output_dir))

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,2),dpi=300,constrained_layout=True,sharex=False)
sns.kdeplot(coor_data_all['back_y'],color='#1E88E5',lw = 5)
sns.kdeplot(coor_data_all['back_y'],color='#1E88E5',lw = 0,fill = True)
plt.title('back_y distribution')
plt.axis('off')
plt.savefig(r'{}/ALL_back_y_distribution_density.png'.format(output_dir))

fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(10,7),dpi=300,constrained_layout=True,sharex=False)
ax[0].plot(X_range,y_list/y_list.max(),c='#1E88E5',lw = 6,zorder=0)
ax[0].scatter(x_smooth[start_point],y_list[int(x_smooth[start_point])]/y_list.max(),marker='^',c='red',ec='black',s=200,zorder=1)
ax[0].text(x_smooth[start_point],y_list[int(x_smooth[start_point])]/y_list.max()+0.3,s = '{:.2f} mm'.format(x_smooth[start_point]))
print('the boundary bwtween perimeter and center:{}'.format(x_smooth[start_point]))
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_linewidth(3)
ax[0].spines['left'].set_linewidth(3)
ax[0].set_xticks([])
#ax2 = ax.twinx()


y_smooth_plot = moving_average(slope1,5)

ax[1].plot(X_range,y_smooth_plot/max(y_smooth_plot),c='#00BCD4',lw = 5,zorder=0)
ax[1].scatter(x_smooth[start_point],y_smooth[start_point]/max(y_smooth),marker='o',c='red',ec='black',s=200,zorder=1)
ax[1].fill(x_smooth[start_point:end_point],(y_smooth[start_point:end_point]/max(y_smooth)),c='#00BCD4',alpha=0.2,zorder=0)

#ax[1].scatter(end_point,(y_smooth[end_point]/max(y_smooth)),marker='o',c='red',ec='black',s=200,zorder=1)
#ax[1].scatter(results_half[2],(y_smooth[int(results_half[2])]/max(y_smooth)),marker='o',c='red',ec='black',s=200,zorder=1)
#ax[1].scatter(results_half[3],(y_smooth[int(results_half[3])]/max(y_smooth)),marker='o',c='red',ec='black',s=200,zorder=1)


ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_linewidth(3)
ax[1].spines['left'].set_linewidth(3)

fig.tight_layout(pad=5.0)
plt.savefig(r'{}/ALL_distribution_density_boundaries.png'.format(output_dir))
