# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 15:37:38 2023

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



FeA_label_file = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\anno_MV_csv-3sThreshold\rec-1-G1-anno_Feature_Space.csv'
cali3d_file = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\BeACSV_origin\rec-1-G1_Cali_Data3d.csv'
para_file = r'D:\Personal_File\yejiailn\3D_spontaneous_data\DayAMLightOn_NightLightOn\result\BeACSV_origin\rec-1-G1_Paras.csv'

cali3d_data = pd.read_csv(cali3d_file)
para_data = pd.read_csv(para_file)


body_part = ['nose','left_ear','right_ear','neck',
             'left_front_limb','right_front_limb','left_hind_limb','right_hind_limb',
             'left_front_claw','right_front_claw','left_hind_claw','right_hind_claw',
             'back','root_tail','mid_tail','tip_tail',
            ]


body_color_dict = {'nose':'#1E2C59',
                   'left_ear':'#192887',
                   'right_ear':'#1B3A95',
                   'neck':'#204FA1',
                   'left_front_limb':'#1974BA',
                   'right_front_limb':'#1EA2BA',
                   'left_hind_limb':'#42B799',
                   'right_hind_limb':'#5CB979',
                   'left_front_claw':'#7BBF57',
                   'right_front_claw':'#9EC036',
                   'left_hind_claw':'#BEAF1F',
                   'right_hind_claw':'#C08719',
                   'back':'#BF5D1C',
                   'root_tail':'#BE3320',
                   'mid_tail':'#9B1F24',
                   'tip_tail':'#6A1517',
                                
    }

movement_color_dict={'running':'#F44336','walking':'#FF5722','left_turning':'#FFAB91','right_turning':'#FFCDD2','stepping':'#BCAAA4',
                     'sniffing':'#26A69A','climb_up':'#43A047','rearing':'#66BB6A','hunching':'#81C784','rising':'#9CCC65','jumping':'#FFB74D',
                     'grooming':'#AB47BC','pause':'#90A4AE',}



def return_AnnoMovement(number):
    movement_dict = {'running':[33,1,2],
                 'trotting':[40,10,9],
                 'walking':[8],
                 'right_turning':[5,30],
                 'left_turning':[6,34],	
                 'climb_up':[39,11,12],	
                 'falling':[27,14],	
                 'rising':[32,13],
                 'grooming':[24,22,21],
                 'down_search':[35,18,7,26,38],	
                 'stepping':[25,19,3],
                 'sniffing':[37,31,4,17],
                 'rearing':[28,23,],
                 'pause':[16,15,36,20,29]
                 }
    for key in movement_dict.keys():
        if number in movement_dict[key]:
            return(key)






event_list = []
start = 0
end = 0
baseline = 0
segBoundary_dict = {}
FeA_file = pd.read_csv(FeA_label_file)
for index in range(60):
    movement_label = FeA_file.loc[index,'movement_label']
    segBoundary_dict.setdefault(movement_label,[])
    end = FeA_file.loc[index,'segBoundary'] + baseline*30*60*10
    lasting_time = end - start + 1
    segBoundary_dict[movement_label].append((start,lasting_time))
    start = end
baseline +=1
event_list.append(segBoundary_dict)

mouse_event = event_list[0]


# =============================================================================
#fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12,10),constrained_layout=True,dpi=300)
# 
# axes[0].plot(para_data.loc[0:end,'speed_back'],lw=3)
# axes[1].plot(para_data.loc[0:end,'movement_energy_nose'],lw=3)
# 
# axes[2].plot(para_data.loc[0:end,'body_height'],lw=3)
# axes[3].plot(para_data.loc[0:end,'body_length'],lw=3)
# axes[4].plot(para_data.loc[0:end,'body_angle'],lw=3)
# 
# for key in mouse_event.keys():# #key = 'grooming'
#      movement_label = key
#      color = movement_color_dict[movement_label]
#      axes[5].broken_barh(mouse_event[key],(0,250),facecolors=color)
# 
# axes[0].axis('off')
# axes[1].axis('off')
# axes[2].axis('off')
# axes[3].axis('off')
# axes[4].axis('off')
# axes[5].axis('off')
# =============================================================================

mouse_event = event_list[0]
fig, axes = plt.subplots(nrows=17, ncols=1, figsize=(10,12),constrained_layout=True,dpi=300)

num = 0
for i in body_part:
    X = i + '_x'
    Y = i + '_x'
    Z = i + '_x'
    Speed = 'speed_' + i
    
    axes[num].plot(para_data.loc[0:end,Speed],c=body_color_dict[i],lw=3)
    axes[num].set_ylim(0,400)
    axes[num].axis('off')
    num += 1
    

for key in mouse_event.keys():
#key = 'grooming'
    movement_label = key
    color = movement_color_dict[movement_label]
    axes[num].broken_barh(mouse_event[key],(0,250),facecolors=color)
    axes[num].set_ylim(0,400)
plt.axis('off')
