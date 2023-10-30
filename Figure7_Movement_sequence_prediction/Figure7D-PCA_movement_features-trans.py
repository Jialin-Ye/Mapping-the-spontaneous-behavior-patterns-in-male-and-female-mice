# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:08:46 2023

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from math import pi


new_generate_data_MOTP_NFDP = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-MOTP_NFDP'           #trans
new_generate_data_NFTP_MODP = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure7_Movement_sequence_prediction\predicted_movement_sequences\new_sequence-NFTP_MODP'
InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure7_Movement_sequence_prediction\PCA_results'

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

def get_path2(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-'))&(file_name.endswith(content)):
            USN = file_name.split('-')[1]
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(USN,date)
            file_path_dict.setdefault(USN,file_dir+'\\'+file_name)
    return(file_path_dict)


Movement_Label_path = get_path(InputData_path_dir,'revised_Movement_Labels.csv')
Movement_Label_path_predict_day = get_path2(new_generate_data_MOTP_NFDP,'Movement_Labels.csv')
Movement_Label_path_predict_night = get_path2(new_generate_data_NFTP_MODP,'Movement_Labels.csv')


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





def extract_segment(df):
    #for index in range(30*seg_size,df.shape[0]+5,30*seg_size):
    mv_dict = {}
    for mv in movement_order:
        mv_dict.setdefault(mv,0)
    count_df = pd.DataFrame(mv_dict,index=[0])

    label_count = df.value_counts('revised_movement_label')
    #movement_label_num, movement_label_type_num = countComplicated(df)
    #center_area_time,enter_center_number = center_count(df) 
    #print(movement_label_num,movement_label_type_num)
    for count_mv in label_count.index:
        count_df[count_mv]  = label_count[count_mv]
    #count_df['movement_label_num'] = movement_label_num
    #count_df['movement_label_type_num'] = movement_label_type_num
    #count_df['center_area_time'] = center_area_time
    #count_df['enter_center_number'] = enter_center_number
    #df_out.reset_index(drop=True,inplace=True)
    return(count_df)




dataset1 = Morning_lightOn_info
dataset2 = Night_lightOff_info

group1 = 'Morning_lightOn'
group2 =  'Night_lightOff'


all_matrix = []

for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    gender =  dataset1.loc[index,'gender']
    ExperimentTime =  dataset1.loc[index,'ExperimentTime']
    LightingCondition = dataset1.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    #location_file = pd.read_csv(Location_path1[video_index],usecols=['location'])
    # MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 0
    MoV_matrix['mouse_info'] = gender+'-' + ExperimentTime+'-' + LightingCondition
    all_matrix.append(MoV_matrix)

for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    gender =  dataset2.loc[index,'gender']
    ExperimentTime =  dataset2.loc[index,'ExperimentTime']
    LightingCondition = dataset2.loc[index,'LightingCondition']
    MoV_file = pd.read_csv(Movement_Label_path[video_index])
    #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 1
    MoV_matrix['mouse_info'] = gender+'-' + ExperimentTime + '-' + LightingCondition
    all_matrix.append(MoV_matrix)

for key in Movement_Label_path_predict_day.keys():
    video_index = key
    gender =  'None'
    ExperimentTime =  group1.split('_')[0]
    LightingCondition =group1.split('_')[1]
    MoV_file = pd.read_csv(Movement_Label_path_predict_day[video_index])
    #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 0
    MoV_matrix['mouse_info'] = 'new_generate_data_MOTP_NFDP'
    all_matrix.append(MoV_matrix)

for key in Movement_Label_path_predict_night.keys():
    video_index = key
    gender =  'None'
    ExperimentTime =  group2.split('_')[0]
    LightingCondition =group2.split('_')[1]
    MoV_file = pd.read_csv(Movement_Label_path_predict_night[video_index])
    #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
    #MoV_file = pd.concat([MoV_file,location_file],axis=1)
    MoV_matrix = extract_segment(MoV_file)
    MoV_matrix['group_id'] = 1
    MoV_matrix['mouse_info'] = 'new_generate_data_NFTP_MODP'
    all_matrix.append(MoV_matrix)

all_df = pd.concat(all_matrix)
all_df.reset_index(drop=True,inplace=True)


feat_cols = all_df.columns[:-2]
# Separating out the features
X = all_df[feat_cols].values
# Separating out the target
y = all_df['group_id']
# Standardizing the features
X = StandardScaler().fit_transform(X)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
PC1 = principalComponents[:,0]
PC2 = principalComponents[:,1]
#PC3 = principalComponents[:,2]

skf = StratifiedKFold(n_splits=2,random_state=2020, shuffle=True)

# split traning dataset and testing dataset
for train_index, test_index in skf.split(X, y):
    #print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)



svm = SVC(kernel='linear')
svm.fit(X_train_pca, y_train)

w = svm.coef_[0]                       
a = -w[0]/w[1]                         
x = np.linspace(-6,6)                  
y = a * x -(svm.intercept_[0])/w[1]    

b = svm.support_vectors_[0]        
y_down = a * x + (b[1] - a*b[0])   
b = svm.support_vectors_[-1]       
y_up = a * x + (b[1] - a*b[0])


fig,ax= plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)


plt.plot(x, y, 'k-', label='SVM Decision Boundary')

plt.plot(x,y_down,'k--')   
plt.plot(x,y_up,'k--')     


cross_vertices = [(-1,-2), # left
             (1,-2), # right
             (0,-1), #top
             (0,-3),] #bottom
 
cross_codes = [mpath.Path.MOVETO,mpath.Path.LINETO,mpath.Path.MOVETO,mpath.Path.LINETO]
cross_path = mpath.Path(cross_vertices, cross_codes)
 
circle = mpath.Path.unit_circle() 
verts = np.concatenate([circle.vertices, cross_path.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, cross_path.codes])
female_marker = mpath.Path(verts, codes)

arrow_vertices = [(0.6,0.6), #end
                   (2.85,2.85), #tip
                   (1.2,2), #top
                   (2.85,2.85),
                   (2,1.2),] #bottom
arrow_codes = [mpath.Path.MOVETO,mpath.Path.LINETO,mpath.Path.LINETO,mpath.Path.MOVETO,mpath.Path.LINETO]
arrow_path = mpath.Path(arrow_vertices, arrow_codes)
verts2 = np.concatenate([circle.vertices, arrow_path.vertices[::-1, ...]])
codes2 = np.concatenate([circle.codes, arrow_path.codes])
male_marker = mpath.Path(verts2, codes2,closed=True)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, all_df[['mouse_info']]], axis = 1)
targets = all_df['mouse_info'].unique()
 
#colors = ['#F5B25E', '#F5B25E', '#A16F3A','#A16F3A']   # day-am & day-pm
#colors = ['#F5B25E', '#F5B25E', '#003960','#003960','#E6E243','#71BFCB']     # day-am & night-lightOFF, MD+MT(True)
colors = ['#F5B25E', '#F5B25E', '#003960','#003960','#279DDA','#DA4A1C']     # day-am & night-lightOFF, MD+MT(Transi)
#colors = ['#3498DB', '#3498DB', '#053143','#053143',]   # lightOn & lightOff
shapes = [female_marker,male_marker,male_marker,female_marker,'^','^']
for target, color,shape in zip(targets,colors,shapes):
     indicesToKeep = finalDf['mouse_info'] == target
     if target.startswith('new_generate_data_MOTP_NFDP'):
         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , ec = 'black'
                    , s = 200
                    , lw = 1
                    ,marker=shape
                    ,alpha=0.9)
     elif target.startswith('new_generate_data_NFTP_MODP'):
         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , ec = 'black'
                    , s = 200
                    , lw = 1
                    ,marker=shape
                    ,alpha=0.9)    
     else:
         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                    , finalDf.loc[indicesToKeep, 'principal component 2']
                    , c = color
                    , ec = 'black'
                    , s = 1500
                    , lw = 2
                    ,marker=shape
                    ,alpha=1)


ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)

plt.xlim(-6,6)
plt.ylim(-10,10)
plt.xticks([])
plt.yticks([])
plt.savefig('{}/MO_TP-NF_DP&MO_TP-NF_DP.png'.format(output_dir),dpi=300)
# Show the plot
plt.show()
