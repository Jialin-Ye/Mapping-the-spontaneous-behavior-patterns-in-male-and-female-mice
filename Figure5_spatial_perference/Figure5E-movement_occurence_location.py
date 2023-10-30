# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:59:28 2023

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


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label" 
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure5_spatial_perference\movement_occurence_location'


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


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']

cluster_color_dict={'locomotion':'#DC2543',                     
                     'exploration':'#009688',
                     'maintenance':'#A13E97',
                     'nap':'#D3D4D4'}

movement_color_dict = {'running':'#FF3030',
                       'trotting':'#F06292',
                       'walking':'#EB6148',
                       'left_turning':'#F6BBC6',
                       'right_turning':'#F29B78',
                       'stepping':'#E4CF7B',                       
                       'jumping':'#ECAD4F',
                       'climbing':'#2E7939',                       
                       'rearing':'#88AF26',
                       'hunching':'#7AB69F',
                       'rising':'#80DEEA',
                       'sniffing':'#2C93CB',                       
                       'grooming':'#A13E97',
                       'pausing':'#D3D4D4',}

location_list = ['center','wall','corner']

location_area = {'center':(18*2)*(18*2),
                 'wall':4*(25-18)*(18*2),
                 'corner':4*(25-18)*(25-18)}



def count_MovbyMin2(df,start_min,end_min):

    df_copy = df.copy()
    start = start_min*60*30
    end = end_min*60*30
    count_df_list = []
    temp_df = df_copy.loc[start:end,:]

    info_dict = {'revised_movement_label':[],'location':[],'count_p':[],'count_divideByarea':[]}
    
    for mv in movement_order:
        mv_df =  temp_df[temp_df['revised_movement_label']==mv]
        for loc in location_list:
            loc_df = mv_df[mv_df['location']==loc]  ## location_word for 16
            if len(mv_df) == 0:
                pass
            else:
                info_dict['revised_movement_label'].append(mv)
                info_dict['location'].append(loc)
                info_dict['count_p'].append((len(loc_df)/len(mv_df)))
                info_dict['count_divideByarea'].append(((len(loc_df)/30)/location_area[loc]))
                
    df_out = pd.DataFrame(info_dict)
    for index in df_out.index:
        mv = df_out.loc[index,'revised_movement_label']
        
        
        value_sum1 = df_out.loc[df_out['revised_movement_label']==mv,'count_p'].sum()
        value_sum2 = df_out.loc[df_out['revised_movement_label']==mv,'count_divideByarea'].sum()
        
        df_out.loc[index,'count_p'] = df_out.loc[index,'count_p'] / value_sum1
        df_out.loc[index,'count_divideByarea_p'] = df_out.loc[index,'count_divideByarea'] / value_sum2

    return(df_out)

def access_loc_mov(dataset): 
    loc_df_list = []
    distance_ratio_df_list = []
    ExperimentTime = dataset['ExperimentTime'].unique()[0]

    
    #gender = dataset['gender'].unique()[0]
    for i in list(dataset.index):
        video_index = dataset.loc[i,'video_index']
        ExperimentTime = dataset.loc[i,'ExperimentTime']
        LightingCondition = dataset.loc[i,'LightingCondition']
        mouse_id = ExperimentTime +'-'+ LightingCondition + '_'+ dataset.loc[i,'mouse_id']  
        mov_data = pd.read_csv(Movement_Label_path[video_index],usecols=['revised_movement_label','back_x','back_y','location'])
        mov_data['back_x'] = mov_data['back_x'] - 250
        mov_data['back_y'] = mov_data['back_y'] - 250
        mov_data['distance_ratio'] =  np.sqrt(np.square(mov_data['back_x']) + np.square(mov_data['back_y'])) / np.sqrt(np.square(250)*2)

        loc_min_df = count_MovbyMin2(mov_data,0,60)      ### min, 1 for 1min, 5 for 5min     

        loc_min_df['mouse_id'] = mouse_id
        loc_min_df['ExperimentTime'] = ExperimentTime
        loc_min_df['gender'] = dataset.loc[i,'gender']
        loc_df_list.append(loc_min_df)
        distance_ratio_df_list.append(mov_data)
    return(loc_df_list,distance_ratio_df_list)


def average_df(df_list,key_word):
    df = pd.concat(df_list)
    df_out = pd.DataFrame()
    for mv in df['revised_movement_label'].unique():
        for loc in df['location'].unique():
            df_out.loc[mv,loc] = df.loc[((df['revised_movement_label']==mv) & (df['location']==loc)),key_word].mean()
    return(df_out)



dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
dataset1_color = '#F5B25E'                      ## morning '#F5B25E' afternoon 936736   night-light-off '#003960'    night-lightOn '#3498DB'
loc_mov_df_list1, distance_ratio_df_list1 = access_loc_mov(dataset1) 
average_forHeatmap_df1 = average_df(loc_mov_df_list1,key_word='count_p') #count_divideByarea_p

dataset2 = Night_lightOn_info
dataset2_name = 'Night_lightOn_info'
loc_mov_df_list2, distance_ratio_df_list2 = access_loc_mov(dataset2) 
average_forHeatmap_df2 = average_df(loc_mov_df_list2,key_word='count_p') #count_divideByarea_p
dataset2_color = '#3498DB'


distance_ratio_df_list1_df = pd.concat(distance_ratio_df_list1)
df_forlineplot1 = pd.DataFrame()
individual_mv_df_list1 = []
num = 0
for mv in movement_order:
    individual_mv_df = distance_ratio_df_list1_df[distance_ratio_df_list1_df['revised_movement_label']==mv]
    df_forlineplot1.loc[num,'revised_movement_label'] = mv
    df_forlineplot1.loc[num,'distance_ratio_mean'] = individual_mv_df['distance_ratio'].mean()
    individual_mv_df_list1.append(individual_mv_df.sample(100,random_state=1))
    num +=1
distance_ratio_sort_df1 = pd.concat(individual_mv_df_list1)

distance_ratio_df_list2_df = pd.concat(distance_ratio_df_list2)
df_forlineplot2 = pd.DataFrame()
individual_mv_df_list2 = []
num = 0
for mv in movement_order:
    individual_mv_df = distance_ratio_df_list2_df[distance_ratio_df_list2_df['revised_movement_label']==mv]
    df_forlineplot2.loc[num,'revised_movement_label'] = mv
    df_forlineplot2.loc[num,'distance_ratio_mean'] = individual_mv_df['distance_ratio'].mean()
    if len(individual_mv_df) > 100:
        individual_mv_df_list2.append(individual_mv_df.sample(100,random_state=1))
    else:
        individual_mv_df_list2.append(individual_mv_df)
    num +=1
distance_ratio_sort_df2 = pd.concat(individual_mv_df_list2)





fig = plt.figure(figsize=(10, 8),dpi=300)
#sns.heatmap(average_forHeatmap_df,vmin=0,vmax=1,square=True,cmap='crest',linewidths=1,cbar=False)
grid = plt.GridSpec(nrows=14, ncols=3,wspace=1)

ax1 = fig.add_subplot(grid[0:14,0:1])
ax2 = fig.add_subplot(grid[0:14,1:2],sharey=ax1)
ax3 = fig.add_subplot(grid[0:14,2:4])

sns.heatmap(average_forHeatmap_df1,vmin=0,vmax=1,square=True,cmap='crest',linewidths=1,ax=ax1,cbar=False)
sns.heatmap(average_forHeatmap_df2,vmin=0,vmax=1,square=True,cmap='crest',linewidths=1,ax=ax2,cbar=False)

ax1.set_title(dataset1_name)
ax2.set_title(dataset2_name)  
#ax1.set_yticks([])
#ax1.set_ylabel('')
#ax2.set_yticks([])
#ax2.set_ylabel('')

ax3.set_title('distance to center')
sns.stripplot(x='distance_ratio',y='revised_movement_label',data=distance_ratio_sort_df1,orient='h',ax=ax3,size=2,palette=movement_color_dict.values(),order=movement_order,edgecolor='black',linewidth=0.1)
ax3.plot(df_forlineplot1['distance_ratio_mean'],df_forlineplot1['revised_movement_label'],lw=3,color=dataset1_color)

sns.stripplot(x='distance_ratio',y='revised_movement_label',data=distance_ratio_sort_df2,orient='h',ax=ax3,size=2,palette=movement_color_dict.values(),order=movement_order,edgecolor='grey',linewidth=0.1)
ax3.plot(df_forlineplot2['distance_ratio_mean'],df_forlineplot2['revised_movement_label'],lw=3,color=dataset2_color)

ax3.set_xlim(0,1)
ax3.set_yticks([])
ax3.set_ylabel('')
ax3.set_xlabel('')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

plt.savefig(r'{}\{}_{}_movement_location.png'.format(output_dir,dataset1_name,dataset2_name),dpi=300)