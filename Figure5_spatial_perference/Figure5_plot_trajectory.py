# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:42:22 2022

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
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.stats as st



output_dir = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\Figure5_spatial_perference\Example_trajectory'

### demo_data
coor_file = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\01_BehaviorAtlas_collated_data\rec-1-MorningLight-on-20220224_normalized_coordinates_back_XY.csv'
Mov_file = r'F:\spontaneous_behavior\GitHub\Elaborate-spontaneous-activity-atlas-driven-behavior-patterns-in-both-sexes\02_revised_movement_label\rec-1-MorningLight-on-20220224_revised_Movement_Labels.csv'

coor_data = pd.read_csv(coor_file,index_col=0)
coor_data['location'] = 'periphery'
coor_data.loc[(coor_data['back_x']>= 125)&(coor_data['back_x']<=375)&(coor_data['back_y']>= 125)&(coor_data['back_y']<=375),'location'] = 'center'


### plot track scatter distributioin (tranditional method)

fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
sns.scatterplot(data=coor_data,x='back_x',y='back_y', ec="none", s=10, hue='location',hue_order=['center','periphery'],
                 legend=False,palette=['#2C7EC2','#BDBDBD'],alpha = 1, ax=ax)

ax.plot([0,0],[0,500],color='black',lw=5)
ax.plot([0,500],[500,500],color='black',lw=5)
ax.plot([500,500],[500,0],color='black',lw=5)
ax.plot([500,0],[0,0],color='black',lw=5)
# 
ax.plot([125,375],[125,125],color='#2C7EC2',lw=5) # tranditional defined center
ax.plot([125,375],[375,375],color='#2C7EC2',lw=5)
ax.plot([125,125],[125,375],color='#2C7EC2',lw=5)
ax.plot([375,375],[125,375],color='#2C7EC2',lw=5)
plt.axis('off')



### plot track scatter distributioin (data-driven center method)

coor_data['data-driven_location'] = 'center'
#coor_data.loc[(coor_data['back_x']>= 125)&(coor_data['back_x']<=375)&(coor_data['back_y']>= 125)&(coor_data['back_y']<=375),'data-driven_location'] = 'tranditional_center'

coor_data.loc[(coor_data['back_x']< 70)&(coor_data['back_y']<= 430)&(coor_data['back_y']>= 70),'data-driven_location'] = 'periphery'
coor_data.loc[(coor_data['back_x']> 430)&(coor_data['back_y']<= 430)&(coor_data['back_y']>= 70),'data-driven_location'] = 'periphery'
coor_data.loc[(coor_data['back_y']< 70)&(coor_data['back_x']<= 430)&(coor_data['back_x']>= 70),'data-driven_location'] = 'periphery'
coor_data.loc[(coor_data['back_y']> 430)&(coor_data['back_x']<= 430)&(coor_data['back_x']>= 70),'data-driven_location'] = 'periphery'

coor_data.loc[(coor_data['back_x']>430) & (coor_data['back_y']> 430),'data-driven_location'] = 'corner'
coor_data.loc[(coor_data['back_x']< 70) & (coor_data['back_y']< 70),'data-driven_location'] = 'corner'
coor_data.loc[(coor_data['back_x']< 70) & (coor_data['back_y']> 430),'data-driven_location'] = 'corner'
coor_data.loc[(coor_data['back_x']> 430) & (coor_data['back_y']< 70),'data-driven_location'] = 'corner'


fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
sns.scatterplot(data=coor_data,x='back_x',y='back_y', ec="none", s=10, hue='data-driven_location',hue_order=['center','periphery','corner'],
                 legend=False,palette=['#A5D2DC','#A5D2DC','#657999'],alpha = 1, ax=ax)

ax.plot([0,0],[0,500],color='black',lw=5)
ax.plot([0,500],[500,500],color='black',lw=5)
ax.plot([500,500],[500,0],color='black',lw=5)
ax.plot([500,0],[0,0],color='black',lw=5)
# 
ax.plot([0,500],[70,70],color='black',lw=5)
ax.plot([0,500],[430,430],color='black',lw=5)
ax.plot([70,70],[0,500],color='black',lw=5)
ax.plot([430,430],[0,500],color='black',lw=5)
plt.axis('off')
plt.savefig('{}/coordinates_disribution_tranditional_center.png'.format(output_dir),dpi=300)

### plot mice postion 2D distributioin 
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),dpi=300)
g = sns.JointGrid(height=10, ratio=5, space=.05,xlim=(0, 500), ylim=(0, 500))
sns.scatterplot(data=coor_data,x='back_x',y='back_y', ec="none", s=10, hue='data-driven_location',hue_order=['center','periphery','corner'],
                legend=False,palette=['#A5D2DC','#A5D2DC','#657999'],alpha = 1, ax=g.ax_joint)
g.set_axis_labels(xlabel='',ylabel='')

g.ax_joint.plot([0,0],[0,500],color='black',lw=5)
g.ax_joint.plot([0,500],[500,500],color='black',lw=5)
g.ax_joint.plot([500,500],[500,0],color='black',lw=5)
g.ax_joint.plot([500,0],[0,0],color='black',lw=5)

g.ax_joint.plot([0,500],[70,70],color='black',lw=5)
g.ax_joint.plot([0,500],[430,430],color='black',lw=5)
g.ax_joint.plot([70,70],[0,500],color='black',lw=5)
g.ax_joint.plot([430,430],[0,500],color='black',lw=5)

g.ax_joint.set_xticklabels([])
g.ax_joint.set_yticklabels([])

sns.kdeplot(data=coor_data,x='back_x', color = '#A5D2DC', fill=False, linewidth=2, ax=g.ax_marg_x)
sns.kdeplot(data=coor_data,x='back_x', fill='#A5D2DC', linewidth=2, ax=g.ax_marg_x)
sns.kdeplot(data=coor_data,y='back_y', color = '#A5D2DC',fill=False,linewidth=2, ax=g.ax_marg_y)
sns.kdeplot(data=coor_data,y='back_y', fill='#A5D2DC',linewidth=2, ax=g.ax_marg_y)
plt.savefig('{}/coordinates_disribution_data-driven_center.png'.format(output_dir),dpi=300)

# Movement trajectory  =============================================================================

Mov_data = pd.read_csv(Mov_file,usecols=['OriginalDigital_label','annotated_movement_label','revised_movement_label','movement_cluster_label','location','locomotion_speed_smooth','back_z'])

#movement_color_dict2 = {'locomotion':'#F94040','exploration':'#0C8766','maintenance':'#914C99','nap':'#D4D4D4',}
movement_color_dict2 = {'locomotion':'#DC2543','exploration':'#009688','maintenance':'#A13E97','nap':'#D3D4D4'}


time_window1 = 0
time_window2 = 60                                          # set the time window you wanna plot
data = pd.concat([Mov_data,coor_data],axis=1)
data_select = data.iloc[time_window1*30*60:time_window2*30*60,:]


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


  
cmap1 = cm.RdBu_r
norm1 = mcolors.Normalize(vmin = data_select['locomotion_speed_smooth'].min(), vmax= data_select['locomotion_speed_smooth'].max())
data_select['speed_color'] = list(cmap1(norm1(data_select['locomotion_speed_smooth'])))


### plot speed_trajectory
fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
ax.plot([0,0],[0,500],color='black',lw=5)
ax.plot([0,500],[500,500],color='black',lw=5)
ax.plot([500,500],[500,0],color='black',lw=5)
ax.plot([500,0],[0,0],color='black',lw=5)

ax.plot([0,500],[70,70],color='black',lw=5)
ax.plot([0,500],[430,430],color='black',lw=5)
ax.plot([70,70],[0,500],color='black',lw=5)
ax.plot([430,430],[0,500],color='black',lw=5)
plt.axis('off')

for i in range(1,data_select.shape[0]):
    x_coor1 = data_select.loc[i-1,'back_x']
    y_coor1 = data_select.loc[i-1,'back_y']
    x_coor2 = data_select.loc[i,'back_x']
    y_coor2 = data_select.loc[i,'back_y']
    #color = movement_color_dict2[data_10min.loc[i,'new_label']]
    color = data_select.loc[i,'speed_color']
    ax.plot([x_coor1,x_coor2],[y_coor1,y_coor2],c=color,alpha =0.9,lw=2,linestyle='solid')

plt.savefig('{}/speed_trajectory.png'.format(output_dir),dpi=300)

### plot_movement trajectory

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)
for c in movement_color_dict.keys():
    color = movement_color_dict[c]
    ax.scatter(data_select.loc[data_select['revised_movement_label']==c,'back_x'].values,data_select.loc[data_select['revised_movement_label']==c,'back_y'].values,color= color,s=5,zorder=0,alpha=0.8)

    ax.plot([0,0],[0,500],color='black',lw=5)
    ax.plot([0,500],[500,500],color='black',lw=5)
    ax.plot([500,500],[500,0],color='black',lw=5)
    ax.plot([500,0],[0,0],color='black',lw=5)
    
    ax.plot([0,500],[70,70],color='black',lw=4)
    ax.plot([0,500],[430,430],color='black',lw=4)
    ax.plot([70,70],[0,500],color='black',lw=4)
    ax.plot([430,430],[0,500],color='black',lw=4)
    
    #ax.plot([125,375],[125,125],color='black',lw=4)
    #ax.plot([125,375],[375,375],color='black',lw=4)
    #ax.plot([125,125],[125,375],color='black',lw=4)
    #ax.plot([375,375],[125,375],color='black',lw=4)
plt.axis('off')
plt.savefig('{}/movement_trajectory.png'.format(output_dir),dpi=300)


### plot_movement cluster trajectory

fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10),constrained_layout=True,dpi=300)

for c in movement_color_dict2.keys():
    color = movement_color_dict2[c]
    ax.scatter(data_select.loc[data_select['movement_cluster_label']==c,'back_x'].values,data_select.loc[data_select['movement_cluster_label']==c,'back_y'].values,color= color,s=5,zorder=0,alpha=0.8)

   
    ax.plot([0,0],[0,500],color='black',lw=5)
    ax.plot([0,500],[500,500],color='black',lw=5)
    ax.plot([500,500],[500,0],color='black',lw=5)
    ax.plot([500,0],[0,0],color='black',lw=5)
    
    ax.plot([0,500],[70,70],color='black',lw=5)
    ax.plot([0,500],[430,430],color='black',lw=5)
    ax.plot([70,70],[0,500],color='black',lw=5)
    ax.plot([430,430],[0,500],color='black',lw=5)
    
plt.axis('off')
plt.savefig('{}/movement_cluster_trajectory.png'.format(output_dir),dpi=300)


# =============================================================================
# fig,ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5),constrained_layout=True,dpi=300)
# for c in movement_color_dict2.keys():
#     color = movement_color_dict2[c]
#     
#     r = data_10min.loc[data_10min['revised_movement_label']==c,'r'].values
#     
#     positions = np.arange(0, 1,0.001)
#     kernel = st.gaussian_kde(r)
#     
#     f = kernel(positions).T *len(r)
#     
#     ax.plot(positions,f,color=color,lw=3)
#     ax.fill(positions,f,color=color,lw=2,alpha=0.1)
#     #sns.kdeplot(data=data_10min.loc[data_10min['revised_movement_label']==c,'r'].values,color=color,ax=ax,lw=3,)
# 
# r = data_10min['r']
# 
# positions = np.arange(0, 1,0.001)
# kernel = st.gaussian_kde(r)
# 
# f = kernel(positions).T *len(r)
# 
# ax.plot(positions,f,color='black',lw=5)
# 
# 
# plt.axis('off')
# 
# #sns.kdeplot(data=data_10min,x='back_x',color='black',ax=ax,lw=5)
# 
# 
# =============================================================================
#plt.axis('off')

# =============================================================================
# for c in movement_color_dict2.keys():
#     color = movement_color_dict2.values()
#     x = data_10min.loc[data_10min['revised_movement_label']==c,'back_x'].values
#     y = data_10min.loc[data_10min['revised_movement_label']==c,'back_y'].values
# 
#     xmin, xmax = -10, 510
#     ymin, ymax = -10, 510
#     
#     
#     # Peform the kernel density estimate
#     xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#     positions = np.vstack([xx.ravel(), yy.ravel()])
#     values = np.vstack([x, y])
#     kernel = st.gaussian_kde(values)
#     f = np.reshape(kernel(positions).T, xx.shape)
#     max_value = np.max(f)
#     min_value = np.min(f)
#     f = (f - min_value)/(max_value-min_value)
#     print(f.min(),f.max())
#     cmap1 = cm.RdBu_r
#     norm1 = mcolors.Normalize(vmin = f.min(), vmax= f.max())
#     colors =  cmap1(norm1(f))
#     
#     fig = plt.figure(figsize=(10,10),dpi=600)
#     
#     ax = fig.add_subplot(111,projection='3d')
#     ax.plot_surface(xx,yy,f,facecolors=color)
#     ax.set_zlim(0,1.2)
#     ax.view_init(60, 35) 
# =============================================================================


