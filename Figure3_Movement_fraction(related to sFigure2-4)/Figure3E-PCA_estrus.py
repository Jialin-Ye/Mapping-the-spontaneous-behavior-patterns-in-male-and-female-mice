# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 22:52:01 2023

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

InputData_path_dir = r"F:\文章\04返修阶段\revised_movement_label"
output_dir = r'F:\文章\04返修阶段\Figure_and_code\Figure3_Movement_fraction\estrus'
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


animal_info_csv = r'F:\文章\04返修阶段\Table_S1_animal_informationV3.csv'             ## 动物信息表位置
animal_info = pd.read_csv(animal_info_csv)
animal_info = animal_info[~animal_info['new_index'].isin(skip_file_list)]

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



female_dataset = Night_lightOn_info[Night_lightOn_info['gender']=='female']
diestrus_info = female_dataset[female_dataset['estrous_cycle']=='diestrus']
estrus_info = female_dataset[female_dataset['estrous_cycle']=='estrus']
metestrus_info = female_dataset[female_dataset['estrous_cycle']=='metestrus']
proestrus_info = female_dataset[female_dataset['estrous_cycle']=='proestrus']


dataset1 = diestrus_info       
dataset2 = estrus_info
dataset3 = metestrus_info       
dataset4 = proestrus_info

dataset_name1 = 'diestrus'
dataset_name2 = 'estrus'
dataset_name3 = 'metestrus'
dataset_name4 = 'proestrus'


all_matrix = []
#for file_name in list(Movement_Label_path.keys())[2:73]:
for index in dataset1.index:
    video_index = dataset1.loc[index,'new_index']
    if video_index in skip_file_list:
        pass
    else:
        gender =  dataset1.loc[index,'gender']
        ExperimentTime =  dataset1.loc[index,'ExperimentTime']
        LightingCondition = dataset1.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path1[video_index],usecols=['location'])
       # MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 0
        MoV_matrix['mouse_info'] = 'diestrus'
        all_matrix.append(MoV_matrix)

for index in dataset2.index:
    video_index = dataset2.loc[index,'new_index']
    if video_index in skip_file_list:
        pass
    else:
        gender =  dataset2.loc[index,'gender']
        ExperimentTime =  dataset2.loc[index,'ExperimentTime']
        LightingCondition = dataset2.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
        #MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 1
        MoV_matrix['mouse_info'] = 'estrus'
        all_matrix.append(MoV_matrix)

for index in dataset3.index:
    video_index = dataset3.loc[index,'new_index']
    if video_index in skip_file_list:
        pass
    else:
        gender =  dataset3.loc[index,'gender']
        ExperimentTime =  dataset3.loc[index,'ExperimentTime']
        LightingCondition = dataset3.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path1[video_index],usecols=['location'])
       # MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 2
        MoV_matrix['mouse_info'] = 'metestrus'
        all_matrix.append(MoV_matrix)

for index in dataset4.index:
    video_index = dataset4.loc[index,'new_index']
    if video_index in skip_file_list:
        pass
    else:
        gender =  dataset4.loc[index,'gender']
        ExperimentTime =  dataset4.loc[index,'ExperimentTime']
        LightingCondition = dataset4.loc[index,'LightingCondition']
        MoV_file = pd.read_csv(Movement_Label_path[video_index])
        #location_file = pd.read_csv(Location_path2[video_index],usecols=['location'])
        #MoV_file = pd.concat([MoV_file,location_file],axis=1)
        MoV_matrix = extract_segment(MoV_file)
        MoV_matrix['group_id'] = 3
        MoV_matrix['mouse_info'] = 'proestrus'
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


# 进行降维
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)


# =============================================================================
# skf = StratifiedKFold(n_splits=2,random_state=2020, shuffle=True)
# #print(skf)
# 
# #做划分是需要同时传入数据集和标签
# for train_index, test_index in skf.split(X, y):
#     #print('TRAIN:', train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# 
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)
# 
# # Transform the training and testing data using PCA
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)
# =============================================================================


PC1 = principalComponents[:,0]
PC2 = principalComponents[:,1]
#PC3 = principalComponents[:,2]


fig,ax= plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=600)

# =============================================================================
# # Step 2: Train an SVM classifier
# svm = SVC(kernel='linear')
# svm.fit(X_train_pca, y_train)
# 
# w = svm.coef_[0]                       #权重向量（二维）  coef_存放回归系数
# a = -w[0]/w[1]                         #直线的斜率
# x = np.linspace(-5,5)                  #在指定间隔（-5,5）之间产生等差数列作为x坐标的值
# y = a * x -(svm.intercept_[0])/w[1]    #intercept_[0]存放截距w3
# 
# b = svm.support_vectors_[0]        #第一个支持向量的点
# y_down = a * x + (b[1] - a*b[0])   #超平面下方的直线的方程
# b = svm.support_vectors_[-1]       #最后一个支持向量点
# y_up = a * x + (b[1] - a*b[0])     #超平面上方的直线的方程
# 
# 
# # Step 5: Plot the decision boundary and margins
# plt.plot(x, y, 'k-', label='SVM Decision Boundary')
# 
# plt.plot(x,y_down,'k--')   #与支持向量相切下面的直线
# plt.plot(x,y_up,'k--')     #与支持向量相切上面的直线
# =============================================================================


cross_vertices = [(-1,-2), # left
             (1,-2), # right
             (0,-1), #top
             (0,-3),] #bottom
 
cross_codes = [mpath.Path.MOVETO,mpath.Path.LINETO,mpath.Path.MOVETO,mpath.Path.LINETO]
cross_path = mpath.Path(cross_vertices, cross_codes)
 
circle = mpath.Path.unit_circle()  # 圆形Path 
 # 整合两个路径对象的点
verts = np.concatenate([circle.vertices, cross_path.vertices[::-1, ...]])
 # 整合两个路径对象点的类型
codes = np.concatenate([circle.codes, cross_path.codes])
 
 # 根据路径点和点的类型重新生成一个新的Path对象 
female_marker = mpath.Path(verts, codes)

arrow_vertices = [(0.6,0.6), #end
                   (2.85,2.85), #tip
                   (1.2,2), #top
                   (2.85,2.85),
                   (2,1.2),] #bottom
arrow_codes = [mpath.Path.MOVETO,mpath.Path.LINETO,mpath.Path.LINETO,mpath.Path.MOVETO,mpath.Path.LINETO]
arrow_path = mpath.Path(arrow_vertices, arrow_codes)
 
verts2 = np.concatenate([circle.vertices, arrow_path.vertices[::-1, ...]])
# 整合两个路径对象点的类型
codes2 = np.concatenate([circle.codes, arrow_path.codes])
male_marker = mpath.Path(verts2, codes2,closed=True)

# # 查看降维后的数据
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, all_df[['mouse_info']]], axis = 1)
targets = all_df['mouse_info'].unique()
 
#colors = ['#F5B25E', '#F5B25E', '#A16F3A','#A16F3A']   # day-am & day-pm
#colors = ['#F5B25E', '#F5B25E', '#003960','#003960']     # day-am & night-lightOFF
colors = ['#C13043', '#E37835','#F2D31E','#8F9296']   # lightOn & lightOff

## 'non-estrus' '#F7D5DF'  'estrus' '#ED83A2'

shapes = [female_marker,female_marker,female_marker,female_marker]
for target, color,shape in zip(targets,colors,shapes):
     indicesToKeep = finalDf['mouse_info'] == target
     # 选择某个label下的数据进行绘制
     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                , finalDf.loc[indicesToKeep, 'principal component 2']
                , c = color
                , ec = 'black'
                , s = 2500
                , lw = 2
                ,marker=shape
                ,alpha=1)



ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)

plt.xlim(-7,7)
plt.ylim(-5,5)
plt.xticks([])
plt.yticks([])

# Show the plot
#plt.axis('off')
plt.savefig('{}/{}&{}_PCA.png'.format(output_dir,dataset_name1,dataset_name2),dpi=600)
plt.show()







