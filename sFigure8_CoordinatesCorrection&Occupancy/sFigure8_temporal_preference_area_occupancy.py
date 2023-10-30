# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:14:38 2023

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
#Speed_distance_path =  



animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]


def add_traditional_center(df):
    df['tranditional_location'] = 'wall'
    df.loc[(df['back_x']>125)&(df['back_x']<375)&(df['back_y']>125)&(df['back_y']>375),'tranditional_location'] = 'center'
    df.loc[(df['back_x']<125)&(df['back_y']<125),'tranditional_location'] = 'corner'
    df.loc[(df['back_x']<125)&(df['back_y']>375),'tranditional_location'] = 'corner'
    df.loc[(df['back_x']>375)&(df['back_y']>375),'tranditional_location'] = 'corner'
    df.loc[(df['back_x']>375)&(df['back_y']<125),'tranditional_location'] = 'corner'
    return(df)


dataset = Night_lightOff_info
dataset_name = 'Night_lightOff'

output_dir = output_dir + '/{}'.format(dataset_name) 
if not os.path.exists(output_dir):                                                                       
    os.mkdir(output_dir)

MV_loc_data_list = []
for index in dataset.index:
    video_index = dataset.loc[index,'video_index']
    gender = dataset.loc[index,'gender']
    
    MV_loc_data = pd.read_csv(Movement_Label_path[video_index])  
    MV_loc_data = add_traditional_center(MV_loc_data)
    MV_loc_data['frame'] = MV_loc_data.index +1
    MV_loc_data_list.append(MV_loc_data)
    
df_combine = pd.concat(MV_loc_data_list,axis=0)
df_combine.reset_index(drop=True,inplace=True)


start = 0
end = 0
location_color_dict = {'center':'#DBEDF4',
                       'wall':'#A5D2DC',
                       'corner':'#657999',}

tranditional_location_color_dict = {'center':'#E87E78',
                                    'wall':'#E85850',
                                    'corner':'#D0241A',}

location_order = location_order = {'center':1,
                  'wall':2,
                  'corner':3}
for i in range(10,61,10):             ## calculate by per 10 min
    end = i
    temp_df = df_combine.loc[(df_combine['frame']>=start*30*60) & (df_combine['frame']<=end*30*60),:]
    count_df = temp_df.value_counts('location').to_frame(name='count')
    count_df['percentage'] = count_df['count'] / count_df['count'].sum()
    count_df['color'] = count_df.index.map(location_color_dict)
    count_df['plot_order'] = count_df.index.map(location_order)
    count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    ax.pie(count_df['percentage'],counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            #autopct= '%1.1f%%',pctdistance=1.2,
            colors = count_df['color'],
            radius = 1,
            wedgeprops = {'linewidth':5, 'edgecolor': "black"},)
            #hatch=['.', '.', '.', '.'])
    #start = end
    plt.savefig(r'{}\{}_{}min_accumulate.png'.format(output_dir,dataset_name,i),dpi=300)

for i in range(10,61,10):             ## calculate by per 10 min
    end = i
    temp_df = df_combine.loc[(df_combine['frame']>=start*30*60) & (df_combine['frame']<=end*30*60),:]
    count_df = temp_df.value_counts('tranditional_location').to_frame(name='count')
    count_df['percentage'] = count_df['count'] / count_df['count'].sum()
    count_df['color'] = count_df.index.map(tranditional_location_color_dict)
    count_df['plot_order'] = count_df.index.map(location_order)
    count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
    ax.pie(count_df['percentage'],counterclock=True,normalize=False,startangle = 90,
            #explode = [0.08,0.08,0.08,0.08],
            #autopct= '%1.1f%%',pctdistance=1.2,
            colors = count_df['color'],
            radius = 1,
            wedgeprops = {'linewidth':5, 'edgecolor': "black"},)
            #hatch=['.', '.', '.', '.'])
    #start = end
    plt.savefig(r'{}\{}_TraditionalCenter_{}min_accumulate.png'.format(output_dir,dataset_name,i),dpi=300)



            
# =============================================================================
# 
# ## category 
# for loc in ['center','wall','corner']:
# #for loc in ['wall']:
#     temp_df_day = df_day_all[df_day_all['location_word']==loc]
#     day_count_df = temp_df_day.value_counts('category').to_frame(name='count')
#     day_count_df['percentage'] = day_count_df['count'] / day_count_df['count'].sum()
#     day_count_df['color'] = day_count_df.index.map(category_color_dict)
#     day_count_df['plot_order'] = day_count_df.index.map(category_order)
#     day_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
#     
#     temp_df_night = df_night_all[df_night_all['location_word']==loc]
#     night_count_df = temp_df_night.value_counts('category').to_frame(name='count')
#     night_count_df['percentage'] = night_count_df['count'] / night_count_df['count'].sum()
#     night_count_df['color'] = night_count_df.index.map(category_color_dict)
#     night_count_df['plot_order'] = night_count_df.index.map(category_order)
#     night_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
#     
#     if len(temp_df_day) >= len(temp_df_night):
#         night_radius = 1
#         day_radius = len(temp_df_day)/len(temp_df_night)
#     else:
#         day_radius = 1
#         night_radius = len(temp_df_night)/len(temp_df_day)
#     
#     #plt.figure(figsize=(10, 10),dpi=300)
#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
#     ax.pie(day_count_df['percentage']/2.01,counterclock=True,normalize=False,startangle = 90,
#             explode = [0.08,0.08,0.08,0.08],
#             colors = day_count_df['color'],
#             radius = day_radius,
#             wedgeprops = {'linewidth':5, 'edgecolor': "#F5B25E"},)
#             #hatch=['.', '.', '.', '.'])
#     plt.pie(night_count_df['percentage']/2.01,counterclock=False,normalize=False,startangle = 90,
#             explode = [0.08,0.08,0.08,0.08],
#             colors = night_count_df['color'],
#             radius = night_radius,
#             wedgeprops = {'linewidth':5, 'edgecolor': "#003960"},)
#             #hatch=['.', '.', '.', '.'])
#     plt.pie([1], colors = ['#ffffff'], radius = 0.5)
#     #plt.title(loc)
# =============================================================================








# =============================================================================
# ### movement
# for sex in ['male','female']:
# #for loc in ['wall']:
#     temp_df_day = df_day_all[df_day_all['gender']==sex]
#     day_count_df = temp_df_day.value_counts('category').to_frame(name='count')
#     day_count_df['percentage'] = day_count_df['count'] / day_count_df['count'].sum()
#     day_count_df['color'] = day_count_df.index.map(category_color_dict)
#     day_count_df['plot_order'] = day_count_df.index.map(category_order)
#     day_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
#     
#     temp_df_night = df_night_all[df_night_all['gender']==sex]
#     night_count_df = temp_df_night.value_counts('category').to_frame(name='count')
#     night_count_df['percentage'] = night_count_df['count'] / night_count_df['count'].sum()
#     night_count_df['color'] = night_count_df.index.map(category_color_dict)
#     night_count_df['plot_order'] = night_count_df.index.map(category_order)
#     night_count_df.sort_values(by=['plot_order'],ascending=True,inplace=True)
#     
#     day_radius = night_radius = 1
#     
# # =============================================================================
# #     if len(temp_df_day) >= len(temp_df_night):
# #         night_radius = 1
# #         day_radius = len(temp_df_day)/len(temp_df_night)
# #     else:
# #         day_radius = 1
# #         night_radius = len(temp_df_night)/len(temp_df_day)
# #     
# # =============================================================================
#     #plt.figure(figsize=(10, 10),dpi=300)
#     fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
#     ax.pie(day_count_df['percentage']/2,counterclock=True,normalize=False,startangle = 90,
#             #explode = [0.08,0.08,0.08,0.08],
#             colors = day_count_df['color'],
#             radius = day_radius,
#             #autopct= '%1.1f%%',pctdistance=1.2,
#             wedgeprops = {'linewidth':3, 'edgecolor': "black"},)
#             #wedgeprops = {'linewidth':3, 'edgecolor': "#F5B25E"},)
#             #hatch=['.', '.', '.', '.'])
#     plt.pie(night_count_df['percentage']/2,counterclock=False,normalize=False,startangle = 90,
#             #explode = [0.08,0.08,0.08,0.08],
#             colors = night_count_df['color'],
#             radius = night_radius,
#             #autopct= '%1.1f%%',pctdistance=1.2,
#             wedgeprops = {'linewidth':3, 'edgecolor': "black"})
#             #wedgeprops = {'linewidth':3, 'edgecolor': "#003960"},)
#             #hatch=['.', '.', '.', '.'])
#     plt.pie([1], colors = ['#ffffff'], radius = 0.5)
#     #plt.savefig(r'D:\Personal_File\yejiailn\课题\文章\图\第二版-figure_code\figure3\category_pie_{}_开光灯.png'.format(sex),transparent=True,dpi=300)
# 
# 
# 
# 
# =============================================================================



