# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 21:40:57 2023

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
from random import sample
import matplotlib.patches as patches
import matplotlib.path as mpath


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure3_Movement_fraction\01_morning&night-lightOff_related_to_Figure3'

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

animal_info_csv = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'             ## 动物信息表位置
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




plot_dataset = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'
plot_dataset_male = plot_dataset[plot_dataset['gender']=='male']
plot_dataset_female = plot_dataset[plot_dataset['gender']=='female']

plot_dataset2 = Night_lightOff_info
dataset2_name = 'Night_lightOff'
plot_dataset2_male = plot_dataset2[plot_dataset2['gender']=='male']
plot_dataset2_female = plot_dataset2[plot_dataset2['gender']=='female']

event_list = []
mouse_id_list = []
night_num = 0
female_num = 0



for video_index in plot_dataset2_male.sample(5)['video_index']:
#for video_index in nightLightOff_info_female['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    #night_num +=1
    female_num += 1

for video_index in plot_dataset2_female.sample(5)['video_index']:
#for video_index in nightLightOff_info_male['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    #night_num +=1



for video_index in plot_dataset_male.sample(5)['video_index']:
#for video_index in nightLightOff_info_female['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    night_num +=1
    female_num += 1

for video_index in plot_dataset_female.sample(5)['video_index']:
#for video_index in nightLightOff_info_male['video_index']:
    start = 0
    end = 0
    baseline = 0
    segBoundary_dict = {}
    FeA_file = pd.read_csv(Feature_space_path[video_index])
    for index in range(FeA_file.shape[0]):
        movement_label = FeA_file.loc[index,'revised_movement_label']
        segBoundary_dict.setdefault(movement_label,[])
        end = FeA_file.loc[index,'segBoundary_end'] + baseline*30*60*10
        lasting_time = end - start + 1
        segBoundary_dict[movement_label].append((start,lasting_time))
        start = end
    baseline +=1
    event_list.append(segBoundary_dict)
    mouse_id_list.append(video_index)
    night_num +=1




plot_list = ['stepping','sniffing','hunching','grooming','rising','climbing']                  ## morniong lightOn vs night lightOff difference movement
plot_list2 = ['grooming']                                                                      ## morning lightOn vs afternoon light difference movement
plot_list3 = ['walking','stepping','hunching','rising','grooming','pausing']        ## night lightOn vs night lightOff  difference movement



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(40,14),constrained_layout=True,dpi=600)
ax.add_patch(
patches.Rectangle(
    (-3000, 0),   # (x,y)
    109000,          # width
    0.55*night_num,          # height
    alpha=0.1,ec='#003960', fc='#003960',lw=0.1,zorder=0)
    )

ax.add_patch(
patches.Rectangle(
    (-3000, 0.55*night_num),   # (x,y)
    110000,          # width
    0.55*night_num,          # height
    alpha=0.1,ec="#F5B25E", fc='#F5B25E',lw=0.1,zorder=0)                                   ### Morning F5B25E   night-lightOFF 003960  afternoon 936736 
    )                                                                                       ### night-lightOn  398FCB
num = 0
for i in range(len(event_list)):
    mouse_event = event_list[i]
    mouse_id = mouse_id_list[i]
    for key in mouse_event.keys():
        ##if key =='grooming':
        if key in plot_list:                                                                ## plot the select movement                                                                   
            movement_label = key
        #category = return_category5(movement_label)
        #category_color = cluster_color_dict[category]
            color = movement_color_dict[movement_label]
            ax.broken_barh(mouse_event[key],(num,0.5),facecolors=color,zorder=1)
        else:
            movement_label = key
            color = 'white'
            ax.broken_barh(mouse_event[key],(num,0.5),facecolors=color,zorder=1)
            #plt.text(-2,num,mouse_id,fontsize=40)

    num +=0.55
plt.hlines(0.55*5,-100,109000,'black',linestyles='dashed',linewidths=3)
plt.hlines(0.55*15,-100,109000,'black',linestyles='dashed',linewidths=3)


plt.vlines(-4000,-0.15,11,'black',linestyles='-',linewidths=6)
plt.hlines(-0.15,-4000,109000,'black',linestyles='-',linewidths=6)
for i in range(7):
    plt.vlines(30*60*10*i,-0.15,-0.3,'black',linestyles='-',linewidths=6)
plt.hlines(0.55*night_num,-100,109000,'black',linestyles='dashed',linewidths=6)
plt.axis("off")

plt.savefig(r'{}\{}_{}.png'.format(output_dir,dataset1_name,dataset2_name),dpi=600, transparent=True)