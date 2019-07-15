#!/usr/bin/env python

import rospy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from collections import deque

import rosbag

k=0
pqr_ax_az=deque()
total_vel=deque()
wind_vel=deque()
pose=deque()

def wind_vel_interpolate(wind_vel, t_approx):

	index=0
#	print('len wind', len(wind_vel))
#	print(t_approx)
#	print((wind_vel[index-1].header.stamp).to_sec())
#	print((wind_vel[index].header.stamp).to_sec())
	while (wind_vel[index].header.stamp).to_sec()>t_approx:
		index=index+1
		if index==len(wind_vel):
			break
#	print('wind id', index)
	if index==0:
        	u_w=wind_vel[0].twist.twist.linear.x
       		v_w=wind_vel[0].twist.twist.linear.y
        	return u_w, v_w;
	if index>=len(wind_vel):
        	u_w=wind_vel[len(wind_vel)-1].twist.twist.linear.x
       		v_w=wind_vel[len(wind_vel)-1].twist.twist.linear.y
        	return u_w, v_w;            
            
	x2=(wind_vel[index-1].header.stamp).to_sec()
	x1=(wind_vel[index].header.stamp).to_sec()
	help_coeff2=x2-x1
            
	u_w=(t_approx*(wind_vel[index-1].twist.twist.linear.x-wind_vel[index].twist.twist.linear.x)+x2*wind_vel[index].twist.twist.linear.x-x1*wind_vel[index-1].twist.twist.linear.x)/help_coeff2
	v_w=(t_approx*(wind_vel[index-1].twist.twist.linear.y-wind_vel[index].twist.twist.linear.y)+x2*wind_vel[index].twist.twist.linear.y-x1*wind_vel[index-1].twist.twist.linear.y)/help_coeff2
            
	return u_w, v_w;
        
def total_velocity_interpolate(total_vel, t_approx):

	index=0
	while (total_vel[index].header.stamp).to_sec()>t_approx:
		index=index+1
		if index==len(total_vel):
			break
#        print('total vel ind',index)    
	if index==0:

        	tot_u_i=total_vel[0].twist.twist.linear.x
        	tot_v_i=total_vel[0].twist.twist.linear.y
        	tot_w_i=total_vel[0].twist.twist.linear.z
		return tot_u_i, tot_v_i, tot_w_i;
	if index>=len(total_vel):
        	tot_u_i=total_vel[len(total_vel)-1].twist.twist.linear.x
        	tot_v_i=total_vel[len(total_vel)-1].twist.twist.linear.y
        	tot_w_i=total_vel[len(total_vel)-1].twist.twist.linear.z
		return tot_u_i, tot_v_i, tot_w_i;	
            
	x2=(total_vel[index-1].header.stamp).to_sec()
	x1=(total_vel[index].header.stamp).to_sec()
	help_coeff2=x2-x1
            
	tot_u_i=(t_approx*(total_vel[index-1].twist.twist.linear.x-total_vel[index].twist.twist.linear.x)+x2*total_vel[index].twist.twist.linear.x-x1*total_vel[index-1].twist.twist.linear.x)/help_coeff2
	tot_v_i=(t_approx*(total_vel[index-1].twist.twist.linear.y-total_vel[index].twist.twist.linear.y)+x2*total_vel[index].twist.twist.linear.y-x1*total_vel[index-1].twist.twist.linear.y)/help_coeff2
	tot_w_i=(t_approx*(total_vel[index-1].twist.twist.linear.z-total_vel[index].twist.twist.linear.z)+x2*total_vel[index].twist.twist.linear.z-x1*total_vel[index-1].twist.twist.linear.z)/help_coeff2
            
	return tot_u_i, tot_v_i, tot_w_i; 
        
def pose_interpolate(pose, t_approx):

	index=0
	while (pose[index].header.stamp).to_sec()>t_approx:
		index=index+1
		if index==len(pose):
			break
#        print('pose ind',index)       
	if index==0:

        	e0=pose[0].pose.orientation.w
        	e1=pose[0].pose.orientation.x
        	e2=pose[0].pose.orientation.y
        	e3=pose[0].pose.orientation.z
       		return e0, e1, e2, e3
	if index>=len(pose):
        	e0=pose[len(pose)-1].pose.orientation.w
        	e1=pose[len(pose)-1].pose.orientation.x
        	e2=pose[len(pose)-1].pose.orientation.y
        	e3=pose[len(pose)-1].pose.orientation.z
       		return e0, e1, e2, e3		
        
	x2=(pose[index-1].header.stamp).to_sec()
	x1=(pose[index].header.stamp).to_sec()
	help_coeff2=x2-x1
            
	e0=(t_approx*(pose[index-1].pose.orientation.w-pose[index].pose.orientation.w)+x2*pose[index].pose.orientation.w-x1*pose[index-1].pose.orientation.w)/help_coeff2
	e1=(t_approx*(pose[index-1].pose.orientation.x-pose[index].pose.orientation.x)+x2*pose[index].pose.orientation.x-x1*pose[index-1].pose.orientation.x)/help_coeff2
	e2=(t_approx*(pose[index-1].pose.orientation.y-pose[index].pose.orientation.y)+x2*pose[index].pose.orientation.y-x1*pose[index-1].pose.orientation.y)/help_coeff2
	e3=(t_approx*(pose[index-1].pose.orientation.z-pose[index].pose.orientation.z)+x2*pose[index].pose.orientation.z-x1*pose[index-1].pose.orientation.z)/help_coeff2    
	return e0, e1, e2, e3;

def pqr_ax_az_interpolate(pqr_ax_az, t_approx):
        
	index=0
	while (pqr_ax_az[index].header.stamp).to_sec()>t_approx:
		index=index+1
		if index==len(pqr_ax_az):
			break
#	print('pqr ind', index)      
	if index==0:
            
            	p=pqr_ax_az[0].angular_velocity.x
            	q=pqr_ax_az[0].angular_velocity.y
            	r=pqr_ax_az[0].angular_velocity.z
            	ax=pqr_ax_az[0].linear_acceleration.x
            	az=pqr_ax_az[0].linear_acceleration.z
				
            	return p, q, r, ax, az;
	
	if index>=len(pqr_ax_az):
		
            	print('Warning: Index for pqr_ax_az_interpolate is greater than length of deque')
            	p=pqr_ax_az[len(pqr_ax_az)-1].angular_velocity.x
            	q=pqr_ax_az[len(pqr_ax_az)-1].angular_velocity.y
            	r=pqr_ax_az[len(pqr_ax_az)-1].angular_velocity.z
            	ax=pqr_ax_az[len(pqr_ax_az)-1].linear_acceleration.x
            	az=pqr_ax_az[len(pqr_ax_az)-1].linear_acceleration.z
			
            	return p, q, r, ax, az;	    
        
	x2=(pqr_ax_az[index-1].header.stamp).to_sec()
	x1=(pqr_ax_az[index].header.stamp).to_sec()
	help_coeff2=x2-x1
        
	p=(t_approx*(pqr_ax_az[index-1].angular_velocity.x-pqr_ax_az[index].angular_velocity.x)+x2*pqr_ax_az[index].angular_velocity.x-x1*pqr_ax_az[index-1].angular_velocity.x)/help_coeff2
	q=(t_approx*(pqr_ax_az[index-1].angular_velocity.y-pqr_ax_az[index].angular_velocity.y)+x2*pqr_ax_az[index].angular_velocity.y-x1*pqr_ax_az[index-1].angular_velocity.y)/help_coeff2
	r=(t_approx*(pqr_ax_az[index-1].angular_velocity.z-pqr_ax_az[index].angular_velocity.z)+x2*pqr_ax_az[index].angular_velocity.z-x1*pqr_ax_az[index-1].angular_velocity.z)/help_coeff2
	ax=(t_approx*(pqr_ax_az[index-1].linear_acceleration.x-pqr_ax_az[index].linear_acceleration.x)+x2*pqr_ax_az[index].linear_acceleration.x-x1*pqr_ax_az[index-1].linear_acceleration.x)/help_coeff2
	az=(t_approx*(pqr_ax_az[index-1].linear_acceleration.z-pqr_ax_az[index].linear_acceleration.z)+x2*pqr_ax_az[index].linear_acceleration.z-x1*pqr_ax_az[index-1].linear_acceleration.z)/help_coeff2
	
	return p, q, r, ax, az;

def prepare_training_sample(total_vel, pose, wind_vel, t_approx):
        
        (u_iw, v_iw, w_iw)=total_velocity_interpolate(total_vel, t_approx)
        (u_w, v_w)=wind_vel_interpolate(wind_vel, t_approx)
        (e0, e1, e2, e3)=pose_interpolate(pose, t_approx)
#	(p, q, r, ax, az)=pqr_ax_az_interpolate(pqr_ax_az, t_approx)
                    
        u_i=u_iw-u_w
        v_i=v_iw-v_w
        w_i=w_iw-0
                    
        u_b=(e0*e0+e1*e1-e2*e2-e3*e3)*u_i+2*(e1*e2+e0*e3)*v_i+2*(e1*e3-e0*e2)*w_i
        v_b=2*(e1*e2-e0*e3)*u_i+(e0*e0-e1*e1+e2*e2-e3*e3)*v_i+2*(e2*e3+e0*e1)*w_i
        w_b=2*(e1*e3+e0*e2)*u_i+2*(e2*e3-e0*e1)*v_i+(e0*e0-e1*e1-e2*e2+e3*e3)*w_i

        V=(u_b*u_b+v_b*v_b+w_b*w_b)**(0.5)

        alpha=np.arctan2(w_b, u_b)
        if alpha>3.14:
            alpha=alpha-2*np.pi

        phi=np.arctan2(2*(e0*e1+e2*e3),1-2*(e1*e1+e2*e2))

        if phi>3.14:
            phi=phi-2*np.pi

        theta=np.arcsin(2*(e0*e2-e1*e3))

        
        return u_b, v_b, w_b, V, alpha, phi, theta;

m=0

with rosbag.Bag('output.bag', 'w') as outbag:
    for topic, msg, t in rosbag.Bag('flexible_penta_2019-02-21-14-56-57.bag').read_messages(topics=["/mavros/imu/data_raw", "/mavros/global_position/local", "/mavros/wind_estimation", "/mavros/local_position/pose"]):
	if t.to_sec()>1550757661:
		if topic=="/mavros/imu/data_raw":
			k=k+1
#			print('usao imu')
			if k%10==7:
				pqr_ax_az.appendleft(msg)
		if topic=="/mavros/global_position/local":
#			print('usao total_vel')
			total_vel.appendleft(msg)
		if topic=="/mavros/local_position/pose":
#			print('usao pose')
			pose.appendleft(msg)
		if topic=="/mavros/wind_estimation":
#			print('usao_wind')
			m=m+1
			wind_vel.appendleft(msg)

print(len(pqr_ax_az))

while pqr_ax_az[len(pqr_ax_az)-1].header.stamp.to_sec()<total_vel[len(total_vel)-1].header.stamp.to_sec() or pqr_ax_az[len(pqr_ax_az)-1].header.stamp.to_sec()<pose[len(pose)-1].header.stamp.to_sec() or pqr_ax_az[len(pqr_ax_az)-1].header.stamp.to_sec()<wind_vel[len(wind_vel)-1].header.stamp.to_sec():
	pqr_ax_az.pop()
	
print(len(pqr_ax_az))

validation_set=np.zeros((len(pqr_ax_az), 3))
validation_set_label=np.zeros((len(pqr_ax_az), 3))

for i in range(len(pqr_ax_az)):
	print('usao u i', i)
	(u_b, v_b, w_b, V, alpha, phi, theta)=prepare_training_sample(total_vel, pose, wind_vel, pqr_ax_az[i].header.stamp.to_sec())
	validation_set[i, 0]=alpha
	validation_set[i, 1]=pqr_ax_az[i].angular_velocity.y
	validation_set[i, 2]=V

	validation_set_label[i, 0]=pqr_ax_az[i].linear_acceleration.x
	validation_set_label[i, 1]=pqr_ax_az[i].linear_acceleration.z
	validation_set_label[i, 2]=pqr_ax_az[i].angular_velocity.y

np.save('validation_set', validation_set)
np.save('validation_set_label', validation_set_label)
print('Finished')	
	
	
		


