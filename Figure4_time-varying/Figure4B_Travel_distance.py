# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 22:12:42 2023

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


InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure4_time-varying\Total_travel_distance'

animal_info_csv =   r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Table_S1_animal_information.csv'
animal_info = pd.read_csv(animal_info_csv)

def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if file_name.startswith('rec-') & file_name.endswith(content):
            USN = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'/'+file_name)
    return(file_path_dict)

speed_distance_path = get_path(InputData_path_dir,'speed&distance.csv')

def cal_distance_perMin(df,index):
    df_copy = df.copy()
    start = 0
    end = 0
    df_output = pd.DataFrame()
    num = 0
    for i in range(1,61):
        end = i * 30* 60
        temp_df = df_copy.iloc[start:end,:]
        distance = temp_df['smooth_speed_distance'].sum() 
        df_output.loc[num,'distance'] = distance
        start = end
        num += 1
    return(df_output)    
    

df_distance = pd.DataFrame()

num = 0
start = 0
end = 60 * 30 * 60

data_perMin_list=[]

for index in animal_info.index:
    ExperimentTime = animal_info.loc[index,'ExperimentTime']
    LightingCondition = animal_info.loc[index,'LightingCondition']
    gender = animal_info.loc[index,'gender']
    video_index = animal_info.loc[index,'video_index']
    data = pd.read_csv(speed_distance_path[video_index])
    data = data.iloc[start:end,:]
    data_perMin = cal_distance_perMin(data,index)
    data_perMin['ExperimentCondition'] = ExperimentTime + LightingCondition
    data_perMin['gender'] = gender
    data_perMin_list.append(data_perMin)
    total_distance = data['smooth_speed_distance'].sum()
    df_distance.loc[num,'ExperimentTime'] = ExperimentTime
    df_distance.loc[num,'LightingCondition'] = LightingCondition
    df_distance.loc[num,'ExperimentCondition'] = ExperimentTime+LightingCondition
    df_distance.loc[num,'gender'] = gender
    df_distance.loc[num,'total_distance'] = total_distance
    num +=1

data_output = pd.concat(data_perMin_list,axis=0)

#### single animal distance information 
data_output['time'] = data_output.index
for EC in data_output['ExperimentCondition'].unique():
    TEMP_df = data_output[data_output['ExperimentCondition']==EC]
    data_name = 'distance'
    temp_list = []
    for time_i in range(0,60):
        df_i = TEMP_df.loc[TEMP_df['time']==time_i,data_name]
        df_i_frame = df_i.to_frame(time_i)
        df_i_frame.reset_index(drop=True,inplace=True)
        temp_list.append(df_i_frame.T)
    df_mv_out = pd.concat(temp_list,axis=0)
    #df_mv_out.to_csv(r'{}\total_distance_perMin2_{}.csv'.format(output_dir))

data_output.to_csv(r'{}\total_distance_perMin.csv'.format(output_dir))
df_distance.to_csv(r'{}\total_distance.csv'.format(output_dir))

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,5),dpi=300)
sns.barplot(data=df_distance,x='total_distance',y='gender',orient='h',hue='ExperimentCondition',)   #ExperimentTime, gender