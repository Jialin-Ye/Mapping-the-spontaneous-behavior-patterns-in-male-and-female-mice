# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:27:06 2023

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



FeA_csv_dir = r'H:\spontaneous_behavior_data\BehaviorAtlas_outputs_data\raw_data'
anno_Mov_csv_dir = r'H:\spontaneous_behavior_data\BehaviorAtlas_outputs_data\revised_movement_label'

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)


FeA_path_dict = get_path(FeA_csv_dir,'Feature_Space.csv')
annoMV_path_dict = get_path(anno_Mov_csv_dir,'revised_Movement_Labels.csv')


skip_file_list = [1,3,28,29,110,122]
animal_info_csv = r'H:\spontaneous_behavior_data\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['video_index'].isin(skip_file_list)]

light_on_info = animal_info[animal_info['LightingCondition']=='Light-on']
Morning_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Morning') & (animal_info['LightingCondition']=='Light-on')]
Afternoon_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Afternoon') & (animal_info['LightingCondition']=='Light-on')]

Night_lightOn_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-on')]
Night_lightOff_info = animal_info[(animal_info['ExperimentTime']=='Night') & (animal_info['LightingCondition']=='Light-off')]



calculated_dataset = Morning_lightOn_info

#select_for_plot = [29,28,13,14,22,23,33,26,19,18,17,27,12,10,31,32,25,16,24,34,37,40,15,3,20,21,7]
select_for_plot = range(40)
def add_movement_label_LightON(df):
    movement_dict = {'running':[29,28,13],                     # > 250 mm/s
                     'trotting':[14,22],                       # > 200 mm/s
                     'walking':[23,33,26,19],             # > 80 mm/s
                     'right_turning':[18,17,2,1],              # > 
                     'left_turning':[27,12],
                     'stepping':[9],                           # >50mm/s
                     'climbing':[31,32,25],	
                     'rearing':[16],
                     'hunching':[24],
                     'rising':[5,8,34],
                     'grooming':[37,40,15],
                     'sniffing':[10,11,30,35,36,6,38,4,3],
                     'pausing':[39,20,21],
                     'jumping':[7],
                     }
    df_copy = df.copy()
    for mv in movement_dict.keys():
        df_copy.loc[df_copy['OriginalDigital_label'].isin(movement_dict[mv]),'revised_movement_label'] = mv
    return(df_copy)

def annoFeA(FeA_data,annoMV_data):
    start = 0
    end = 0
    for i in FeA_data.index:
        end = FeA_data.loc[i,'segBoundary']
        single_annoMV_data = annoMV_data.loc[start:end,'revised_movement_label'].value_counts()
        FeA_new_label = single_annoMV_data.index[0]
        start = end
        FeA_data.loc[i,'revised_movement_label'] = FeA_new_label
    return(FeA_data)


movement_order = ['running','trotting','walking','right_turning','left_turning','stepping',
                  'jumping','climbing','rearing','hunching','rising','sniffing',
                  'grooming','pausing']

info_dict = {'revised_movement_label':[],'intra_coff_mean':[],'inter_coff_mean':[]}

for video_index in calculated_dataset['video_index']:
    FeA_data = pd.read_csv(FeA_path_dict[video_index])
    FeA_data = FeA_data[FeA_data['OriginalDigital_label'].isin(select_for_plot)]
    annoMV_data = pd.read_csv(annoMV_path_dict[video_index])
    FeA_data = annoFeA(FeA_data,annoMV_data)
    #FeA_data = add_movement_label_LightON(FeA_data)
      
    for mv in movement_order:
    #for mv in ['jumpping']:
        intra_coff = []
        inter_coff = []
        
        sub_distMatIntra = FeA_data[FeA_data['revised_movement_label']==mv]
        sub_distMatIntra.reset_index(drop=True,inplace=True)
        
        sub_distMatInter = FeA_data[FeA_data['revised_movement_label']!=mv]
        sub_distMatInter.reset_index(drop=True,inplace=True)
        
        if len(sub_distMatIntra) >5 and len(sub_distMatInter) > 5:
            for i in sub_distMatIntra.sample(5,random_state=1).index:
                xx = np.array(sub_distMatIntra.loc[i,['umap1','umap2','zs_velocity']]).astype(float)
                for j in sub_distMatIntra.sample(5,random_state=2).index:
                    yy = np.array(sub_distMatIntra.loc[j,['umap1','umap2','zs_velocity']]).astype(float)
                    tem_coff = np.corrcoef(xx, yy)
                    intra_coff.append(tem_coff[0][1])
                    
                for k in sub_distMatInter.sample(5,random_state=3).index:
                    zz = np.array(sub_distMatInter.loc[k,['umap1','umap2','zs_velocity']]).astype(float)
                    tem_coff = np.corrcoef(xx, zz)
                    inter_coff.append(tem_coff[0][1])
        
            info_dict['revised_movement_label'].append(mv)
            info_dict['intra_coff_mean'].append(np.mean(intra_coff))
            info_dict['inter_coff_mean'].append(np.mean(inter_coff))
        else:
            info_dict['revised_movement_label'].append(mv)
            info_dict['intra_coff_mean'].append(np.nan)
            info_dict['inter_coff_mean'].append(np.nan)


df = pd.DataFrame(info_dict)


new_df = pd.DataFrame()

ouput_dict = {}
for k in df['revised_movement_label'].unique():
    intra_coff_data_column = k + '_intra'
    inter_coff_data_column = k + '_inter'
    ouput_dict.setdefault(intra_coff_data_column,df.loc[df['revised_movement_label']==k,'intra_coff_mean'].values)
    ouput_dict.setdefault(inter_coff_data_column,df.loc[df['revised_movement_label']==k,'inter_coff_mean'].values)

df_output = pd.DataFrame(ouput_dict)
df_output.to_csv(r'H:\spontaneous_behavior_data\Figure&panel_code\Figure1_ExpeimentDesign&DataProcessing\movement_correlation\coff_select.csv')



num = 0
for l in df.index:
    for m in  df.columns[1:]: 
        new_df.loc[num,'revised_movement_label'] =  df.loc[l,'revised_movement_label']
        new_df.loc[num,'group'] =  m
        new_df.loc[num,'value'] =  df.loc[l,m]
        num += 1

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,5),dpi=300)
sns.violinplot(x = 'revised_movement_label', y='value',hue='group',data=new_df,order=movement_order,cut=-0.5,scale_hue=True,bw='silverman',width=0.9,)
plt.xticks(rotation=70)


# =============================================================================
# intra_coff_data = np.zeros((len(x_order),72), dtype = float, order = 'C')
# inter_coff_data = np.zeros((len(x_order),72), dtype = float, order = 'C')
# 
# def adjacent_values(vals, q1, q3):
#     upper_adjacent_value = q3 + (q3 - q1) * 1.5
#     upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
#     lower_adjacent_value = q1 - (q3 - q1) * 1.5
#     lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
#     return lower_adjacent_value, upper_adjacent_value
# 
# def set_axis_style(ax, labels):
#     ax.get_xaxis().set_tick_params(direction='out')
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_xticks(np.arange(1, len(labels) + 1))
#     ax.set_xticklabels(labels)
#     ax.set_xlim(0.25, len(labels) + 0.75)
#     ax.set_xlabel('Sample name')
# 
# num = 0
# for i in df['revised_movement_label'].unique():
#     intra_coff_data[num][:] = df.loc[df['revised_movement_label']==i,'intra_coff_mean'].values
#     inter_coff_data[num][:] = df.loc[df['revised_movement_label']==i,'intra_coff_mean'].values
#     num+=1
#     #inter_coff_data.append(list(df.loc[df['revised_movement_label']==i,'inter_coff_mean'].values))
#     
# 
# 
# fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(20,5),dpi=300)   
# parts = ax.violinplot(intra_coff_data.T,showmeans=False, showmedians=False,showextrema=False,)
# 
# for pc in parts['bodies']:
#     pc.set_facecolor('#D43F3A')
#     pc.set_edgecolor('black')
#     pc.set_alpha(1)
#     
# quartile1, medians, quartile3 = np.percentile(intra_coff_data, [25, 50, 75], axis=1)
# whiskers = np.array([adjacent_values(sorted_array, q1, q3)
#                      
# for sorted_array, q1, q3 in zip(intra_coff_data, quartile1, quartile3)])
# 
# whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]
# inds = np.arange(1, len(medians) + 1)
# ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
#     
# 
# set_axis_style(ax, x_order)
# plt.subplots_adjust(bottom=0.15, wspace=0.05)
#     
# =============================================================================
    
    
    
    
    
    
    
    
    
