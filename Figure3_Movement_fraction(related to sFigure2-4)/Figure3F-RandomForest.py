# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 23:30:20 2023

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold

InputData_path_dir =  r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure3_Movement_fraction\03_night-lightOn&night-lightOff_related_to_sFigure3'


def get_path(file_dir,content):
    file_path_dict = {}
    for file_name in os.listdir(file_dir):
        if (file_name.startswith('rec-')) & (file_name.endswith(content)):
            video_index = int(file_name.split('-')[1])
            #date = i.split('-')[3][0:8]
            #file_name = 'rec-{0}-G1-{1}'.format(video_index,date)
            file_path_dict.setdefault(video_index,file_dir+'\\'+file_name)
    return(file_path_dict)

Movement_Label_path = get_path(InputData_path_dir,'Movement_Labels.csv')



skip_file_list = [1,3,28,29,110,122] 

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




dataset1 = Night_lightOn_info
dataset2 = Night_lightOff_info

dataset_name1 = 'Night_lightOn'
dataset_name2 =  'Night_lightOff'

dateset1_actual_number =0
all_matrix = []
#for file_name in list(Movement_Label_path.keys())[2:73]:
for index in dataset1.index:
    video_index = dataset1.loc[index,'video_index']
    if video_index in skip_file_list:
        pass
    else:
        dateset1_actual_number += 1
        gender =  dataset1.loc[index,'gender']
        ExperimentTime =  dataset1.loc[index,'ExperimentTime']
        LightingCondition = dataset1.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path1[video_index],usecols=['location'])
        #MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 0
        MoV_matrix['mouse_info'] = gender+'-' + ExperimentTime+'-' + LightingCondition
        all_matrix.append(MoV_matrix)

dateset2_actual_number = 0
for index in dataset2.index:
    video_index = dataset2.loc[index,'video_index']
    if video_index in skip_file_list:
        pass
    else:
        dateset2_actual_number += 1
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

print('dateset1_actual_number:',dateset1_actual_number)
print('dateset2_actual_number:',dateset2_actual_number)

all_df = pd.concat(all_matrix)
all_df.reset_index(drop=True,inplace=True)

feat_cols = all_df.columns[:-2]
X = all_df[feat_cols].values
y = all_df['group_id']
X = StandardScaler().fit_transform(X)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 6)
skf = StratifiedKFold(n_splits=2,random_state=2020, shuffle=True)


for train_index, test_index in skf.split(X, y):
    #print('TRAIN:', train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


print('traning_dataset_number:',len(X_train))
print('testing_dataset_number:',len(X_test))


clf = RandomForestClassifier(n_estimators=10, random_state=42,bootstrap=True,criterion='gini',class_weight='balanced',max_depth=100,min_samples_split=2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = clf.score(X_test, y_test)

print(f"Accuracy: {accuracy}")


cm_matrix = confusion_matrix(y_test, y_pred)
print(cm_matrix)


class_names=[0,1] # name  of classes
fig, ax = plt.subplots(figsize=(10,10),dpi=600)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.text(x=0.2,y=-.1,s='Random forest prediction, Accuracy:{:.2f}'.format(accuracy_score(y_test, y_pred)),fontsize=26)
# create heatmap
sns.heatmap(pd.DataFrame(cm_matrix),annot=True, cmap="copper",cbar=False,annot_kws={"fontsize":30})
ax.set_xticklabels([dataset_name1,dataset_name2],fontsize=20)
ax.set_yticklabels([dataset_name1,dataset_name2],fontsize=20)
ax.xaxis.set_label_position("top")
plt.tight_layout()
#plt.axis('off')
plt.title('Confusion matrix', y=1.1,fontsize=40)
plt.ylabel('Actual label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
print('Accuracy:',accuracy_score(y_test, y_pred))
plt.savefig('{}/{}&{}_RandomForest_prediction.png'.format(output_dir,dataset_name1,dataset_name2),dpi=600)
