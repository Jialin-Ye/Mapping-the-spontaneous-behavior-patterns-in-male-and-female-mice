# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 13:43:26 2023

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
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as font_manager
import scipy.stats as stats
import matplotlib.patches as patches

InputData_path_dir = r"F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label"
output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\sFigure6-correlation_between_movement&time\correlation_MV&time'


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

dataset1 = Morning_lightOn_info
dataset1_name = 'Morning_lightOn'

dataset2 = Night_lightOff_info
dataset2_name = 'Night_lightOff'

def movement_label_count(df,time_segement):
    start = 0
    end = 0

    count_df_list = []
    num = 0
    for i in range(time_segement*60*30,df.shape[0]+5,time_segement*60*30):
        num += 1*time_segement
        mv_dict = {}
        for mv in movement_order:
            mv_dict.setdefault(mv,0)
        count_df = pd.DataFrame(mv_dict,index=[num])
        
        end = i
        temp_df = df.loc[start:end,:]
        label_count = temp_df.value_counts('revised_movement_label')
        
        for count_mv in label_count.index:
            count_df[count_mv]  = label_count[count_mv]
        count_df_list.append(count_df)
       
        start = end
    combine_df = pd.concat(count_df_list,axis=0)
    return(combine_df)

def calculate_columns_percentage(df):
    df_copy = df.copy()
    for i in df_copy.columns:
        df_copy[i] = round((df_copy[i] / df_copy[i].sum())*100,2)
    return(df_copy)


def plot(df1,df2):
    fig, axes = plt.subplots(nrows = df1.shape[1],ncols=1,sharex=True,figsize=(15,10),dpi=300)
    
    for i in range(df1.shape[1]):
        x1 = range(df1.shape[0])
        y1 = df1.iloc[:,i]
        label1 = df1.columns[i]
        
        x2 = range(df2.shape[0])
        y2 = df2.iloc[:,i]
        label2 = df2.columns[i]
        
        axes[i].bar(x = x1,height = y1,bottom = 10,color='#F5B25E')
        axes[i].bar(x = x2,height = -y2,bottom = 10,color='#003960')
        #axes[i].set_ylim(0,10)
        axes[i].text(-10,10,label1,fontsize = 15,fontfamily='arial')
        axes[i].set_yticks([])
        axes[i].axis('off')


def average_df(df_list):
    df1 = df_list[0]
    df1 = df1.fillna(0)
    for i in range(1,len(df_list)):      
        df2 = df_list[i]
        df2 = df2.fillna(0)
        df1 = df1 + df2

        #for k in df1.columns:
        #    df1[k] = np.mean([df1[k],df2[k]], 0)
    df1 = df1/len(df_list)
    
    return(df1)


dataset1_percentage_count_df_list = []
for i in dataset1.index:

    video_index = dataset1.loc[i,'video_index']
    ExperimentTime = dataset1_name
    gender = dataset1.loc[i,'gender']
        
    data = pd.read_csv(Movement_Label_path[video_index])
    count_df = movement_label_count(data,1)                          ### min, 1 for 1min, 5 for 5min
    percentage_count_df = calculate_columns_percentage(count_df)
    percentage_count_df.fillna(0)
    dataset1_percentage_count_df_list.append(percentage_count_df)
         
dataset1 = average_df(dataset1_percentage_count_df_list)


dataset2_percentage_count_df_list = []
for i in dataset2.index:
    video_index = dataset2.loc[i,'video_index']
    ExperimentTime = dataset2_name
    gender = dataset2.loc[i,'gender']        
    data = pd.read_csv(Movement_Label_path[video_index])
    count_df = movement_label_count(data,1)           ### min, 1 for 1min, 5 for 5min
    percentage_count_df = calculate_columns_percentage(count_df)
    percentage_count_df.fillna(0)
    dataset2_percentage_count_df_list.append(percentage_count_df)
         
dataset2 = average_df(dataset2_percentage_count_df_list)


font = font_manager.FontProperties(family='arial',
                                   weight='bold',
                                   style='normal', size=12)

for i in dataset1.columns:
    new_df = pd.DataFrame()
    new_df[i] = dataset1[i]
    new_df['time'] = dataset1.index
    
    r1,p1 = stats.pearsonr(new_df['time'],new_df[i])    ### r = relativeship p = significance
    
    exam_X = new_df.loc[:,'time']
    exam_y = new_df.loc[:,i]
    X_train,X_test,y_train,y_test = train_test_split(exam_X, exam_y, train_size = .8)
    X_train = X_train.values.reshape(-1,1)
    X_test = X_test.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(X_train,y_train)
    a = model.intercept_
    b = model.coef_
    r = model.score(X_test,y_test)
    
    night_df = pd.DataFrame()
    night_df[i] = dataset2[i]
    night_df['time'] = dataset2.index
    
    r2,p2 = stats.pearsonr(night_df['time'],night_df[i])    ### r = relativeship p = significance
    
    #df_corr = new_df.corr()
    #p = df_corr.iloc[0,1]
    night_exam_X = night_df.loc[:,'time']
    night_exam_y = night_df.loc[:,i]
    night_X_train, night_X_test, night_y_train, night_y_test = train_test_split(night_exam_X, night_exam_y, train_size = .8)
    night_X_train = night_X_train.values.reshape(-1,1)
    night_X_test = night_X_test.values.reshape(-1,1)
    model_night = LinearRegression()
    model_night.fit(night_X_train,night_y_train)
    a_night = model.intercept_
    b_night = model.coef_
    r_night = model.score(night_X_test,night_y_test)
    
    
    
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,5),layout="constrained",dpi=300)     ## 8:5
    

    y_train_pred = model.predict(X_train)
    y_train_pred_night = model_night.predict(night_X_train)

    plt.plot(X_train,y_train_pred,color='#F5B25E',linewidth=3,zorder=1)
    plt.plot(night_X_train,y_train_pred_night,color='#003960',linewidth=3,linestyle = 'solid',zorder=1)
    
    plt.scatter(X_train,y_train,color='#F5B25E',label='MorningLightOn',ec='black',zorder=2)
    plt.scatter(night_X_train,night_y_train,color='#003960',label='NightLightOff',ec='black',zorder=2)
    

    #plt.annotate('Correlation coefficient',xy=(0.35, 1.2), xycoords='axes fraction',color='#F5B25E',size=18,fontfamily='arial',fontweight="bold")
    if abs(r1) >0.5:
        plt.annotate('r = {:.2f}'.format(r1),xy=(0.35, 1.12), xycoords='axes fraction',color='red',size=20,fontfamily='arial',fontweight="bold")
    elif (abs(r1) > 0.3) & (abs(r1)<=0.5):  
        plt.annotate('r = {:.2f}'.format(r1),xy=(0.35, 1.12), xycoords='axes fraction',color='#827717',size=20,fontfamily='arial',fontweight="bold")
    else:
        plt.annotate('r = {:.2f}'.format(r1),xy=(0.35, 1.12), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    
    if p1 < 0.01:
        plt.annotate('p < 0.01'.format(round(p1,2)),xy=(0.53,1.12), xycoords='axes fraction',color='#4527A0',size=20,fontfamily='arial',fontweight="bold")
    elif (p1 >= 0.01) & (p1 < 0.05):
        plt.annotate('p = {:.2f}'.format(round(p1,2)),xy=(0.53, 1.12), xycoords='axes fraction',color='#4527A0',size=20,fontfamily='arial',fontweight="bold")
    else:
        plt.annotate('p = {:.2f}'.format(round(p1,2)),xy=(0.53, 1.12), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    #plt.annotate('Coefficient of determination',xy=(0.35, 1.1), xycoords='axes fraction',color='#F5B25E',size=20,fontfamily='arial',fontweight="bold")
    plt.annotate('{} = {:.2f}'.format(r'$R^{2}$',r),xy=(0.35, 1.04), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    

    #plt.annotate('Correlation coefficient',xy=(0.7, 1.2), xycoords='axes fraction',color='#003960',size=20,fontfamily='arial',fontweight="bold")
    #plt.annotate('r = {:.2}'.format(r2),xy=(0.65, 1.15), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    
    if abs(r2) >0.5:
        plt.annotate('r = {:.2f}'.format(r2),xy=(0.7, 1.12), xycoords='axes fraction',color='red',size=20,fontfamily='arial',fontweight="bold")
    elif (abs(r2) > 0.3) & (abs(r2)<=0.5):  
        plt.annotate('r = {:.2f}'.format(r2),xy=(0.7, 1.12), xycoords='axes fraction',color='#827717',size=20,fontfamily='arial',fontweight="bold")
    else:
        plt.annotate('r = {:.2f}'.format(r2),xy=(0.7, 1.12), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    
    if p2 < 0.01:
        plt.annotate('p < 0.01'.format(round(p2,2)),xy=(0.88, 1.12), xycoords='axes fraction',color='#4527A0',size=20,fontfamily='arial',fontweight="bold")
    elif (p2 >= 0.01) & (p2 < 0.05):
        plt.annotate('p = {:.2f}'.format(round(p2,2)),xy=(0.88, 1.12), xycoords='axes fraction',color='#4527A0',size=20,fontfamily='arial',fontweight="bold")
    else:
        plt.annotate('p = {:.2f}'.format(round(p2,2)),xy=(0.88, 1.12), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    #plt.annotate('Coefficient of determination',xy=(0.7, 1.1), xycoords='axes fraction',color='#003960',size=18,fontfamily='arial',fontweight="bold")
    plt.annotate('{} = {:.2f}'.format(r'$R^{2}$',r_night),xy=(0.7,  1.04), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    
    
    plt.annotate('(min)',xy=(1, -0.07), xycoords='axes fraction',color='black',size=20,fontfamily='arial',fontweight="bold")
    
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    

    
    plt.ylim(0,5)
    plt.xticks(size=25,fontfamily='arial',fontweight="bold")
    plt.yticks(size=25,fontfamily='arial',fontweight="bold")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0,prop=font)
    #plt.xlabel("Time (min)",fontfamily='arial',fontweight="bold",size=15)
    plt.ylabel("Distribution percentage (%)",fontfamily='arial',fontweight="bold",size=20)

    plt.title(i.title(),fontfamily='arial',fontweight="bold",fontsize=25,pad=15,loc='left')
    #plt.annotate(i.title(),xy=(-0.02, 1.1), xycoords='axes fraction',color='black',size=25,fontfamily='arial',fontweight="bold")
    plt.savefig('{}/{}_{}_{}_time_variation.png'.format(output_dir,dataset1_name,dataset2_name,i))
    plt.show()





















