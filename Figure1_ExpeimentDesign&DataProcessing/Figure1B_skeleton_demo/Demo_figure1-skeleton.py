#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 09:22:20 2022

@author: Jialin-Ye
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import gaussian_kde
import mpl_scatter_density
import os 

output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure1_ExpeimentDesign&DataProcessing\Figure1B_skeleton_demo'

new_ske_file = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data\rec-1-MorningLight-on-20220224_normalized_skeleton_XYZ.csv'
mov_file = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label\rec-1-MorningLight-on-20220224_revised_Movement_Labels.csv'


ske_data = pd.read_csv(new_ske_file,index_col=0)
Mov_data = pd.read_csv(mov_file,usecols=['revised_movement_label'])
conbime_data = pd.concat([Mov_data,ske_data],axis=1)

#select_data = conbime_data[conbime_data['new_label'].isin(select_list)]
select_data = conbime_data.iloc[1188:1327+51,:]                                 ### select plot time window


def plot_3DsideView_skeleton(df_select,i):
#### 3D###
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

    df_singleLocomotion = df_select
    df_singleLocomotion.reset_index(drop=True,inplace=True)
    df_singleLocomotion.loc[len(df_singleLocomotion.index)] = df_singleLocomotion.mean(axis=0)
    
    
    nose = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'nose_z']]
    left_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_ear_z']]
    right_ear = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_ear_z']]
    neck = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'neck_z']]
    left_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_limb_z']]
    right_front_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_limb_z']]
    left_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_limb_z']]
    right_hind_limb = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_limb_z']]
    left_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_front_claw_z']]
    right_front_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_front_claw_z']]
    left_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'left_hind_claw_z']]
    right_hind_claw = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'right_hind_claw_z']]
    back = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'back_z']]
    root_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'root_tail_z']]
    mid_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'mid_tail_z']]
    tip_tail = [df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_x'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_y'],df_singleLocomotion.loc[len(df_singleLocomotion.index)-1,'tip_tail_z']]

    fig = plt.figure(figsize=(8,8),constrained_layout=True,dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(nose[0],nose[1],nose[2],s=60,c=body_color_dict['nose'],alpha = 0.5)
    ax.scatter(left_ear[0],left_ear[1],left_ear[2],s=60,c=body_color_dict['left_ear'],alpha = 0.5)
    ax.scatter(right_ear[0],right_ear[1],right_ear[2],s=60,c=body_color_dict['right_ear'],alpha = 0.5)
    ax.plot([nose[0],left_ear[0]],[nose[1],left_ear[1]],[nose[2],left_ear[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([nose[0],right_ear[0]],[nose[1],right_ear[1]],[nose[2],right_ear[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_ear[0],right_ear[0]],[left_ear[1],right_ear[1]],[left_ear[2],right_ear[2]],c='black',alpha = 0.3,lw=5)
    
    ax.scatter(neck[0],neck[1],neck[2],s=60,c=body_color_dict['neck'],alpha = 0.5)
    ax.scatter(back[0],back[1],back[2],s=60,c=body_color_dict['back'],alpha = 0.5)
    ax.scatter(root_tail[0],root_tail[1],root_tail[2],s=60,c=body_color_dict['root_tail'],alpha = 0.5)
    ax.scatter(mid_tail[0],mid_tail[1],mid_tail[2],s=60,c=body_color_dict['mid_tail'],alpha = 0.5)
    #ax.scatter(tip_tail[0],tip_tail[1],s=40,c='orange')
    
    ax.scatter(left_front_limb[0],left_front_limb[1],left_front_limb[2],s=60,c='black',alpha = 0.5)
    ax.scatter(right_front_limb[0],right_front_limb[1],right_front_limb[2],s=60,c='black',alpha = 0.5)
    ax.scatter(left_hind_limb[0],left_hind_limb[1],left_hind_limb[2],s=60,c='black',alpha = 0.5)
    ax.scatter(right_hind_limb[0],right_hind_limb[1],right_hind_limb[2],s=60,c='black',alpha = 0.5)
    
    ax.scatter(left_front_claw[0],left_front_claw[1],left_front_claw[2],s=60,c=body_color_dict['left_front_claw'],marker='o',ec = 'black',alpha = 0.5)
    ax.scatter(right_front_claw[0],right_front_claw[1],right_front_claw[2],s=60,c=body_color_dict['right_front_claw'],marker='o',ec = 'black',alpha = 0.5)
    ax.scatter(left_hind_claw[0],left_hind_claw[1],left_hind_claw[2],s=60,c=body_color_dict['left_hind_claw'],marker='o',ec = 'black',alpha = 0.5)
    ax.scatter(right_hind_claw[0],right_hind_claw[1],right_hind_claw[2],s=60,c=body_color_dict['right_hind_claw'],marker='o',ec = 'black',alpha = 0.5)
    
    ax.plot([left_front_limb[0],left_hind_limb[0]],[left_front_limb[1],left_hind_limb[1]],[left_front_limb[2],left_hind_limb[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_hind_limb[0],right_hind_limb[0]],[left_hind_limb[1],right_hind_limb[1]],[left_hind_limb[2],right_hind_limb[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_hind_limb[0],right_front_limb[0]],[right_hind_limb[1],right_front_limb[1]],[right_hind_limb[2],right_front_limb[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_front_limb[0],left_front_limb[0]],[right_front_limb[1],left_front_limb[1]],[right_front_limb[2],left_front_limb[2]],c='black',alpha = 0.5,lw=5)
    
    ax.plot([left_ear[0],neck[0]],[left_ear[1],neck[1]],[left_ear[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_ear[0],neck[0]],[right_ear[1],neck[1]],[right_ear[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_front_limb[0],neck[0]],[left_front_limb[1],neck[1]],[left_front_limb[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_front_limb[0],neck[0]],[right_front_limb[1],neck[1]],[right_front_limb[2],neck[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([left_hind_limb[0],root_tail[0]],[left_hind_limb[1],root_tail[1]],[left_hind_limb[2],root_tail[2]],c='black',alpha = 0.5,lw=5)
    ax.plot([right_hind_limb[0],root_tail[0]],[right_hind_limb[1],root_tail[1]],[right_hind_limb[2],root_tail[2]],c='black',alpha = 0.5,lw=5)
    
    ax.plot([root_tail[0],mid_tail[0]],[root_tail[1],mid_tail[1]],[root_tail[2],mid_tail[2]],c='black',alpha = 0.5,lw=5)
    #ax.plot([mid_tail[0],tip_tail[0]],[mid_tail[1],tip_tail[1]],c='black')
    

    ax.plot([left_front_limb[0],back[0]],[left_front_limb[1],back[1]],[left_front_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_front_limb[0],back[0]],[right_front_limb[1],back[1]],[right_front_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([left_hind_limb[0],back[0]],[left_hind_limb[1],back[1]],[left_hind_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_hind_limb[0],back[0]],[right_hind_limb[1],back[1]],[right_hind_limb[2],back[2]],c='black',alpha = 0.5,lw=3)
    
    ax.plot([left_front_limb[0],left_front_claw[0]],[left_front_limb[1],left_front_claw[1]],[left_front_limb[2],left_front_claw[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_front_limb[0],right_front_claw[0]],[right_front_limb[1],right_front_claw[1]],[right_front_limb[2],right_front_claw[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([left_hind_limb[0],left_hind_claw[0]],[left_hind_limb[1],left_hind_claw[1]],[left_hind_limb[2],left_hind_claw[2]],c='black',alpha = 0.5,lw=3)
    ax.plot([right_hind_limb[0],right_hind_claw[0]],[right_hind_limb[1],right_hind_claw[1]],[right_hind_limb[2],right_hind_claw[2]],c='black',alpha = 0.5,lw=3)
    
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_zlim(-50,150)

    ax.view_init(20, 160)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(False)
    plt.axis('off')
    #ax.scatter(df_singleLocomotion.loc[:,'left_front_claw_x'], df_singleLocomotion.loc[:,'left_front_claw_y'], df_singleLocomotion.loc[:,'left_front_claw_z'],color = '#FF1744',alpha=0.2,s=0.5)
    #ax.scatter(df_singleLocomotion.loc[:,'right_front_claw_x'], df_singleLocomotion.loc[:,'right_front_claw_y'],df_singleLocomotion.loc[:,'right_front_claw_z'], color = '#F57C00',alpha=0.2,s=0.5)
    #ax.scatter(df_singleLocomotion.loc[:,'left_hind_claw_x'], df_singleLocomotion.loc[:,'left_hind_claw_y'], df_singleLocomotion.loc[:,'left_hind_claw_z'],color ='#00E676',alpha=0.2,s=0.5)
    #ax.scatter(df_singleLocomotion.loc[:,'right_hind_claw_x'], df_singleLocomotion.loc[:,'right_hind_claw_y'], df_singleLocomotion.loc[:,'right_hind_claw_z'],color ='#00E5FF',alpha=0.2,s=0.5)
    #plt.savefig('{}/{}.png'.format(output_dir,i),dpi=300,transparent=True)



start =0
end = 0
for i in range(0,len(select_data),10):
    end = i
    temp_df = select_data.iloc[start:end,:]
    plot_3DsideView_skeleton(temp_df,i)
    start = end


