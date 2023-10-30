# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:47:55 2023

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
from scipy import stats
from matplotlib import colors

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\sFigure9-Movement_distribution_heatmap\movement_heatmap'


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
#Speed_distance_path =  



animal_info_csv =  r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']


movement_color_dict = {'running':'#DD2C00',
                       'trotting':'#EC407A',
                       'walking':'#FF5722',
                       'left_turning':'#FFAB91',
                       'right_turning':'#FFCDD2',
                       'stepping':'#BCAAA4',
                       'sniffing':'#26A69A',
                       'climbing':'#43A047',
                       'rearing':'#66BB6A',
                       'hunching':'#0288D1',
                       'rising':'#9CCC65',
                       'jumping':'#FFB74D',
                       'grooming':'#AB47BC',
                       'pausing':'#90A4AE',} 

location_list = ['center','wall','corner']

location_area = {'center':(18*2)*(18*2),
                 'wall':4*(25-18)*(18*2),
                 'corner':4*(25-18)*(25-18)}


dataset = Morning_lightOn_info
dataset_info = 'Morning_lightOn'
sub_dir =  output_dir+'\{}'.format(dataset_info)

if not os.path.exists(sub_dir):                                                          
    os.mkdir(sub_dir)   


MoV_file_list = []
for file_name in list(dataset['video_index']):
    MoV_file = pd.read_csv(Movement_Label_path[file_name])
    #MoV_file = add_category(MoV_file)
    #Loc_file = pd.read_csv(Location_path2[file_name],index_col=0)
    #df = MoV_file[['new_label','category4','category6']]
    #df = pd.concat([df,Loc_file],axis=1)
    MoV_file_list.append(MoV_file)
df_all = pd.concat(MoV_file_list,axis=0)


def plot(df,mv):
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10),dpi=300)
    sns.kdeplot(df['back_x'], df['back_y'], shade=True,color=movement_color_dict[mv])

def plot2(df):
    fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
    x = df['back_x']
    y = df['back_y']
    xmin = 0
    xmax = 500
    ymin = 0
    ymax = 500
    X, Y = np.mgrid[xmin:xmax:250j, ymin:ymax:250j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    

    #ax[0].scatter(df['back_x'],df['back_y'],s=1,c='blue')
    #ax[0].set_title('normalized coordinates')
    ax.imshow(np.rot90(Z), cmap='RdBu_r',interpolation='gaussian',extent=[xmin, xmax, ymin, ymax],zorder=0)
    #ax.set_title('heatmap')
    
    
    

    ax.plot((500,0), (0,0),c='black',lw=3,zorder=2)
    ax.plot((0,0), (0,500),c='black',lw=3,zorder=2)
    ax.plot((500,0), (500,500),c='black',lw=3,zorder=2)
    ax.plot((500,500), (500,0),c='black',lw=3,zorder=2)
    #ax.plot((0,0), (500,0),c='black',lw=3,zorder=2)
    ax.axis('equal')
    
    #plt.suptitle(file_name)
    ax.set_xlim(0,500)
    ax.set_ylim(0,500)
    plt.axis('off')
    #plt.xlim(-real_radius-50,real_radius+50)
    #plt.ylim(-real_radius-50,real_radius+50)
    plt.savefig('{}/{}_{}_DensityMovementHeatmap.png'.format(sub_dir,dataset_info,mv),dpi=300)
   # plt.savefig('{}/{}_normalized_coordinates_heatmap.png'.format(output_dir, file_name),dpi=300) 


def plot1(df):
    x = df['back_x']
    y = df['back_y']
    #grid_count = np.histogram2d(x,y,bins=100)                                   ## calculate cell number in each 400 um2 grid
    #df_count = pd.DataFrame(grid_count[0])
    #print(np.quantile(df_count.values,0.25,interpolation='lower'))
    fig,ax = plt.subplots(nrows=1, ncols=1,figsize=(10,10),dpi=300,constrained_layout=True)
    im = ax.hist2d(x,y,bins=100,range=[[0,500], [0,500]],cmap='RdBu_r')#,vmin=0,vmax=600)
    
    #plt.colorbar(im[3])
    plt.axis('off')
    #sns.kdeplot(temp_df['back_x'], temp_df['back_y'], shade=True,color=movement_color_dict2[mv])
    #ax.set_title('{}'.format(mv),family='arial',color='black', weight='normal', size = 40)
    plt.savefig('{}/{}_{}_movementHeatmap.png'.format(sub_dir,dataset_info,mv),dpi=300)
    #df_count.to_csv(output_dir+'/{}_heatmap_data4category.csv'.format(mv))
    #print('{}:{}'.format(mv,len(temp_df)/len(df_all)))

for mv in df_all['revised_movement_label'].unique():
#for mv in ['jumping']:
    temp_df = df_all[df_all['revised_movement_label']==mv]
    plot1(temp_df)

    #plot2(temp_df)
    
    