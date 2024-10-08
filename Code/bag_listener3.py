#!/usr/bin/env python

import rospy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.utils.tensorboard
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from collections import deque

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Temperature
from sensor_msgs.msg import FluidPressure
from nav_msgs.msg import Odometry
from mavros_msgs.msg import ActuatorControl

class DataBuffer():
    
    def wind_velocity_cb(self, data1):
        if not self.processing:
            data1.header.stamp=rospy.get_rostime()
            self.wind_vel.appendleft(data1)
        
    def total_velocity_cb(self, data2):
        if not self.processing:
            data2.header.stamp=rospy.get_rostime()
            self.total_vel.appendleft(data2)
            
    def pose_cb(self, data3):
        if not self.processing:
            data3.header.stamp=rospy.get_rostime()
            self.pose.appendleft(data3)
            
    def static_pressure_cb(self, data4):
        if not self.processing:
            data4.header.stamp=rospy.get_rostime()
            self.static_p.appendleft(data4)
    
    def diff_pressure_cb(self, data5):
        if not self.processing:
            data5.header.stamp=rospy.get_rostime()
            self.diff_p.appendleft(data5)
            
    def temperature_cb(self, data6):
        if not self.processing:
            data6.header.stamp=rospy.get_rostime()
            self.temp.appendleft(data6)
            
    def pqr_and_lin_acc_cb(self, data7):
        if not self.processing:
            data7.header.stamp=rospy.get_rostime()
            self.pqr_ax_az.appendleft(data7)
    
    def odometry_cb(self, data8):
        if not self.processing:
            data8.header.stamp=rospy.get_rostime()
            self.odom.appendleft(data8)

    def elevator_cb(self, data9):
        if not self.processing:
            data9.header.stamp=rospy.get_rostime()
            self.elevator.appendleft(data9)

    def rpm_cb(self, data10):
        if (self.processing==False):
            data10.header.stamp=rospy.get_rostime()
            self.rpm.appendleft(data10)	    
   
    def __init__(self, batch_size=5, frequency=3, updatetime=0.15):
        
        self.processing=False
        self.wind_vel=deque(maxlen=30000)
        self.total_vel=deque(maxlen=30000)
        self.pose=deque(maxlen=30000)
        self.static_p=deque(maxlen=30000)
        self.diff_p=deque(maxlen=30000)
        self.temp=deque(maxlen=30000)
        self.pqr_ax_az=deque(maxlen=60000)
        self.odom=deque(maxlen=30000)
        self.time=deque(maxlen=30000)
        self.elevator=deque(maxlen=30000)	
        self.rpm=deque(maxlen=30000)
        
        self.train_batch_size=batch_size
        self.Ts=updatetime
        
        self.training_set=None
        self.training_label=None
        
        self.training_set_with_thrust=None
        self.training_label_with_thrust=None       
        self.f=frequency
	self.t_0=time.time()
        
        rospy.Subscriber("/mavros/global_position/local", Odometry, self.total_velocity_cb)
        rospy.Subscriber("/mavros/wind_estimation", TwistWithCovarianceStamped, self.wind_velocity_cb)
        rospy.Subscriber("/mavros/imu/data_raw", Imu, self.pqr_and_lin_acc_cb)
        rospy.Subscriber("/mavros/imu/static_pressure", FluidPressure, self.static_pressure_cb)
        rospy.Subscriber("/mavros/imu/diff_pressure", FluidPressure, self.diff_pressure_cb)
        rospy.Subscriber("/mavros/imu/temperature_imu", Temperature, self.temperature_cb)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_cb)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odometry_cb)
        rospy.Subscriber("/mavros/actuator_control", ActuatorControl, self.elevator_cb)
#	rospy.Subscriber("/mavros/actuator_control", ActuatorControl, self.rpm_cb)
             
        rospy.sleep(10*self.Ts)
    
    def wind_vel_interpolate(self, t_approx):

        index=0
        while (self.wind_vel[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.wind_vel):
		break

        if index==0:

            u_w=self.wind_vel[0].twist.twist.linear.x
            v_w=self.wind_vel[0].twist.twist.linear.y
            return u_w, v_w;
        
        if index==len(self.wind_vel):

            u_w=self.wind_vel[len(self.wind_vel)-1].twist.twist.linear.x
            v_w=self.wind_vel[len(self.wind_vel)-1].twist.twist.linear.y
            return u_w, v_w; 
   
 	self.Processing=True           
        x2=(self.wind_vel[index-1].header.stamp).to_sec()
        x1=(self.wind_vel[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
 	
            
        u_w=(t_approx*(self.wind_vel[index-1].twist.twist.linear.x-self.wind_vel[index].twist.twist.linear.x)+x2*self.wind_vel[index].twist.twist.linear.x-x1*self.wind_vel[index-1].twist.twist.linear.x)/help_coeff2
        v_w=(t_approx*(self.wind_vel[index-1].twist.twist.linear.y-self.wind_vel[index].twist.twist.linear.y)+x2*self.wind_vel[index].twist.twist.linear.y-x1*self.wind_vel[index-1].twist.twist.linear.y)/help_coeff2
	self.Processing=False
            
        return u_w, v_w;
        
    def total_velocity_interpolate(self, t_approx):

        index=0
        while (self.total_vel[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.total_vel):
		break
            
        if index==0:

            tot_u_i=self.total_vel[0].twist.twist.linear.x
            tot_v_i=self.total_vel[0].twist.twist.linear.y
            tot_w_i=self.total_vel[0].twist.twist.linear.z
            return tot_u_i, tot_v_i, tot_w_i;

        if index==len(self.total_vel):

            tot_u_i=self.total_vel[len(self.total_vel)-1].twist.twist.linear.x
            tot_v_i=self.total_vel[len(self.total_vel)-1].twist.twist.linear.y
            tot_w_i=self.total_vel[len(self.total_vel)-1].twist.twist.linear.z
            return tot_u_i, tot_v_i, tot_w_i;

	self.Processing=True            
        x2=(self.total_vel[index-1].header.stamp).to_sec()
        x1=(self.total_vel[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
            
        tot_u_i=(t_approx*(self.total_vel[index-1].twist.twist.linear.x-self.total_vel[index].twist.twist.linear.x)+x2*self.total_vel[index].twist.twist.linear.x-x1*self.total_vel[index-1].twist.twist.linear.x)/help_coeff2
        tot_v_i=(t_approx*(self.total_vel[index-1].twist.twist.linear.y-self.total_vel[index].twist.twist.linear.y)+x2*self.total_vel[index].twist.twist.linear.y-x1*self.total_vel[index-1].twist.twist.linear.y)/help_coeff2
        tot_w_i=(t_approx*(self.total_vel[index-1].twist.twist.linear.z-self.total_vel[index].twist.twist.linear.z)+x2*self.total_vel[index].twist.twist.linear.z-x1*self.total_vel[index-1].twist.twist.linear.z)/help_coeff2
	self.Processing=False
            
        return tot_u_i, tot_v_i, tot_w_i; 
        
    def pose_interpolate(self, t_approx):

        index=0
        while (self.pose[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.pose):
		break
               
        if index==0:

            e0=self.pose[0].pose.orientation.w
            e1=self.pose[0].pose.orientation.x
            e2=self.pose[0].pose.orientation.y
            e3=self.pose[0].pose.orientation.z
            return e0, e1, e2, e3

        if index==len(self.pose):

            e0=self.pose[len(self.pose)-1].pose.orientation.w
            e1=self.pose[len(self.pose)-1].pose.orientation.x
            e2=self.pose[len(self.pose)-1].pose.orientation.y
            e3=self.pose[len(self.pose)-1].pose.orientation.z
            return e0, e1, e2, e3

	self.Processing=True        
        x2=(self.pose[index-1].header.stamp).to_sec()
        x1=(self.pose[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
            
        e0=(t_approx*(self.pose[index-1].pose.orientation.w-self.pose[index].pose.orientation.w)+x2*self.pose[index].pose.orientation.w-x1*self.pose[index-1].pose.orientation.w)/help_coeff2
        e1=(t_approx*(self.pose[index-1].pose.orientation.x-self.pose[index].pose.orientation.x)+x2*self.pose[index].pose.orientation.x-x1*self.pose[index-1].pose.orientation.x)/help_coeff2
        e2=(t_approx*(self.pose[index-1].pose.orientation.y-self.pose[index].pose.orientation.y)+x2*self.pose[index].pose.orientation.y-x1*self.pose[index-1].pose.orientation.y)/help_coeff2
        e3=(t_approx*(self.pose[index-1].pose.orientation.z-self.pose[index].pose.orientation.z)+x2*self.pose[index].pose.orientation.z-x1*self.pose[index-1].pose.orientation.z)/help_coeff2
	self.Processing=False
    
        return e0, e1, e2, e3;
        
    def static_pressure_interpolate(self, t_approx):

        index=0
        while (self.static_p[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.static_p):
		break
       
        if index==0:

            s_p=self.static_p[0].fluid_pressure
            return s_p

        if index==len(self.static_p):

            s_p=self.static_p[len(self.static_p)-1].fluid_pressure
            return s_p

	self.Processing=True        
        x2=(self.static_p[index-1].header.stamp).to_sec()
        x1=(self.static_p[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
        
        s_p=(t_approx*(self.static_p[index-1].fluid_pressure-self.static_p[index].fluid_pressure)+x2*self.static_p[index].fluid_pressure-x1*self.static_p[index-1].fluid_pressure)/help_coeff2
	self.Processing=False
        
        return s_p
    
    def diff_pressure_interpolate(self, t_approx):

        index=0
        while (self.diff_p[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.diff_p):
		break
		
        
        if index==0:

            d_p=self.diff_p[0].fluid_pressure
            return d_p

        if index==len(self.diff_p):

            d_p=self.diff_p[len(self.diff_p)-1].fluid_pressure
            return d_p

	self.Processing=True        
        x2=(self.diff_p[index-1].header.stamp).to_sec()
        x1=(self.diff_p[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
        
        d_p=(t_approx*(self.diff_p[index-1].fluid_pressure-self.diff_p[index].fluid_pressure)+x2*self.diff_p[index].fluid_pressure-x1*self.diff_p[index-1].fluid_pressure)/help_coeff2
	self.Processing=False 
       
        return d_p
    
    def temperature_interpolate(self, t_approx):        

        index=0
        while (self.temp[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.temp):
		break
	       
        if index==0:
            t=self.temp[0].temperature
            return t

        if index==len(self.temp):
            t=self.temp[len(self.temp)-1].temperature
            return t

	self.Processing=True        
        x2=(self.temp[index-1].header.stamp).to_sec()
        x1=(self.temp[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
        
        t=(t_approx*(self.temp[index-1].temperature-self.temp[index].temperature)+x2*self.temp[index].temperature-x1*self.temp[index-1].temperature)/help_coeff2
	self.Processing=False

        return t
     
    def pqr_ax_az_interpolate(self, t_approx):
        
        index=0
        while (self.pqr_ax_az[index].header.stamp).to_sec()>t_approx:
            index=index+1
	    if index==len(self.pqr_ax_az):
		break
	      
        if index==0:
            
            p=self.pqr_ax_az[0].angular_velocity.x
            q=self.pqr_ax_az[0].angular_velocity.y
            r=self.pqr_ax_az[0].angular_velocity.z
            ax=self.pqr_ax_az[0].linear_acceleration.x
            az=self.pqr_ax_az[0].linear_acceleration.z
			
            return p, q, r, ax, az;
	
        if index==len(self.pqr_ax_az):

            p=self.pqr_ax_az[len(self.pqr_ax_az)-1].angular_velocity.x
            q=self.pqr_ax_az[len(self.pqr_ax_az)-1].angular_velocity.y
            r=self.pqr_ax_az[len(self.pqr_ax_az)-1].angular_velocity.z
            ax=self.pqr_ax_az[len(self.pqr_ax_az)-1].linear_acceleration.x
            az=self.pqr_ax_az[len(self.pqr_ax_az)-1].linear_acceleration.z
			
            return p, q, r, ax, az;	    

	self.Processing=True        
        x2=(self.pqr_ax_az[index-1].header.stamp).to_sec()
        x1=(self.pqr_ax_az[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
        
        p=(t_approx*(self.pqr_ax_az[index-1].angular_velocity.x-self.pqr_ax_az[index].angular_velocity.x)+x2*self.pqr_ax_az[index].angular_velocity.x-x1*self.pqr_ax_az[index-1].angular_velocity.x)/help_coeff2
        q=(t_approx*(self.pqr_ax_az[index-1].angular_velocity.y-self.pqr_ax_az[index].angular_velocity.y)+x2*self.pqr_ax_az[index].angular_velocity.y-x1*self.pqr_ax_az[index-1].angular_velocity.y)/help_coeff2
        r=(t_approx*(self.pqr_ax_az[index-1].angular_velocity.z-self.pqr_ax_az[index].angular_velocity.z)+x2*self.pqr_ax_az[index].angular_velocity.z-x1*self.pqr_ax_az[index-1].angular_velocity.z)/help_coeff2
        ax=(t_approx*(self.pqr_ax_az[index-1].linear_acceleration.x-self.pqr_ax_az[index].linear_acceleration.x)+x2*self.pqr_ax_az[index].linear_acceleration.x-x1*self.pqr_ax_az[index-1].linear_acceleration.x)/help_coeff2
        az=(t_approx*(self.pqr_ax_az[index-1].linear_acceleration.z-self.pqr_ax_az[index].linear_acceleration.z)+x2*self.pqr_ax_az[index].linear_acceleration.z-x1*self.pqr_ax_az[index-1].linear_acceleration.z)/help_coeff2
	self.Processing=False
	
        return p, q, r, ax, az;
    
    def odometry_interpolate(self, t_approx):

        index=0
        while (self.odom[index].header.stamp).to_sec()>t_approx:
            index=index+1
	        
        if index==0:
            odom_u_i=self.odom[0].twist.twist.linear.x
            odom_v_i=self.odom[0].twist.twist.linear.y
            odom_w_i=self.odom[0].twist.twist.linear.z
            return odom_u_i, odom_v_i, odom_w_i;
            
        x2=(self.odom[index-1].header.stamp).to_sec()
        x1=(self.odom[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)
            
        odom_u_i=(t_approx*(self.odom[index-1].twist.twist.linear.x-self.odom[index].twist.twist.linear.x)+x2*self.odom[index].twist.twist.linear.x-x1*self.odom[index-1].twist.twist.linear.x)/help_coeff2
        odom_v_i=(t_approx*(self.odom[index-1].twist.twist.linear.y-self.odom[index].twist.twist.linear.y)+x2*self.odom[index].twist.twist.linear.y-x1*self.odom[index-1].twist.twist.linear.y)/help_coeff2
        odom_w_i=(t_approx*(self.odom[index-1].twist.twist.linear.z-self.odom[index].twist.twist.linear.z)+x2*self.odom[index].twist.twist.linear.z-x1*self.odom[index-1].twist.twist.linear.z)/help_coeff2
            
        return odom_u_i, odom_v_i, odom_w_i;

    def elevator_interpolate(self, t_approx):
 
        index=0
        while (self.elevator[index].header.stamp).to_sec()>t_approx:
            index=index+1
	     
        if index==0:
            elevator=self.elevator[0].controls[1]
            return elevator
            
        x2=(self.elevator[index-1].header.stamp).to_sec()
        x1=(self.elevator[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)

        elevator=(t_approx*(self.elevator[index-1].controls[1]-self.elevator[index].controls[1])+x2*self.elevator[index].controls[1]-x1*self.elevator[index-1].controls[1])/help_coeff2                
        
        return elevator 
	
    def rpm_interpolate(self, t_approx):
	
        index=0
        while (self.rpm[index].header.stamp).to_sec()>t_approx:
            index=index+1
	   
        if index==0:
            rpm=self.rpm[0].controls[3]
            return rpm
            
        x2=(self.rpm[index-1].header.stamp).to_sec()
        x1=(self.rpm[index].header.stamp).to_sec()
        help_coeff2=x2-x1
	if help_coeff2==0:
		print(index)
		print(x2)
		print(x1)

        rpm=(t_approx*(self.rpm[index-1].controls[3]-self.rpm[index].controls[3])+x2*self.rpm[index].controls[3]-x1*self.rpm[index-1].controls[3])/help_coeff2

        return rpm

    def prepare_training_set(self, model, t_k, t_k_1):

        dT=(t_k-t_k_1)/(1+self.f)
        
        train_batch=np.zeros((self.train_batch_size, model.n_inputs_br2))
        train_label=np.zeros((self.train_batch_size, model.n_outputs_br1+model.n_outputs_br2+3))
        
        for i in range(self.train_batch_size):
            
            self.processing=True
            t_approx=t_k-(i+1)*dT
            (p, q, r, ax, az)=self.pqr_ax_az_interpolate(t_approx)        
            (u_b, v_b, w_b, V, alpha, phi, theta, ro)=self.prepare_training_sample(t_approx)    
            q_dot=self.central_diff_q_dot(dT, t_approx)
            (p_k_plus_1, q_k_plus_1, r_k_plus_1, ax_k_plus_1, az_k_plus_1)=self.pqr_ax_az_interpolate(t_approx+dT)
            if model.Elevator_signal_enabled==True:
                train_batch[i, model.n_inputs_br2-1]=self.elevator_interpolate(t_approx)                  	
            self.processing=False
            
            train_batch[i, 0]=alpha            
            train_batch[i, 1]=q            
            train_batch[i, 2]=V    
 
            train_label[i, 0]=ax
            train_label[i, 1]=az    	
            train_label[i, 2]=q_dot
            train_label[i, 3]=q_k_plus_1
            train_label[i, 4]=q
            train_label[i, 5]=dT
		
            training_set=torch.tensor(train_batch, dtype=torch.float32)            
            training_label=torch.tensor(train_label, dtype=torch.float32)
            
        return training_set, training_label;

    def prepare_training_sample(self, t_approx):
        
        (u_iw, v_iw, w_iw)=self.total_velocity_interpolate(t_approx)
        (u_w, v_w)=self.wind_vel_interpolate(t_approx)
        (e0, e1, e2, e3)=self.pose_interpolate(t_approx)
        s_p=self.static_pressure_interpolate(t_approx)
        d_p=self.diff_pressure_interpolate(t_approx)
        temp=self.temperature_interpolate(t_approx)
        (odom_u_i, odom_v_i, odom_w_i)=self.odometry_interpolate(t_approx)
                    
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
        ro=(s_p+d_p)/(temp+273)/287.058

        
        return u_b, v_b, w_b, V, alpha, phi, theta, ro;  

    def prepare_training_batch_thrust(self, model, t_k, t_k_1):

        dT=(t_k-t_k_1)/(1+self.f)
        
        if model.Elevator_signal_enabled:
            train_batch=np.zeros((self.train_batch_size, model.n_inputs_br2))
        else:
            train_batch=np.zeros((self.train_batch_size, model.n_inputs_br2-1))
            
        train_batch_thrust=np.zeros((self.train_batch_size, model.n_inputs_thrust))
        train_label_thrust=np.zeros((self.train_batch_size, model.n_outputs_thrust))
        
        for i in range(self.train_batch_size):
            
            self.processing=True
            t_approx=t_k-(i+1)*dT
            (p, q, r, ax_t, az_t)=self.pqr_ax_az_interpolate(t_approx)
            (u_b, v_b, w_b, V, alpha, phi, theta, ro)=self.prepare_training_sample(t_approx)
	
            rpm=self.rpm_interpolate(t_approx)
            if model.Elevator_signal_enabled==True:
                train_batch[i, model.n_inputs_br2-1]=self.elevator_interpolate(t_approx)
            self.processing=False
            
            train_batch[i, 0]=alpha
            train_batch[i, 1]=q
            train_batch[i, 2]=V
        
            (o1, o2)=model.forward(torch.tensor(train_batch[i], dtype=torch.float32))
            
            train_batch_thrust[i, 0]=V           
            train_batch_thrust[i, 1]=rpm
            
            train_label_thrust[i, 0]=ax_t-o1.data.to_list()[0]
            train_label_thrust[i, 1]=az_t-o1.data.to_list()[1]

            training_set_with_thrust=torch.tensor(train_batch_thrust, dtype=torch.float32)
            training_label_with_thrust=torch.tensor(train_label_thrust, dtype=torch.float32)
        
        return training_set_with_thrust, training_label_with_thrust;

    def central_diff_q_dot(self, dT, t_approx):
        
        (p_k_1, q_k_1, r_k_1, ax_k_1, az_k_1)=self.pqr_ax_az_interpolate(t_approx-dT)
        (p_kplus1, q_kplus1, r_kplus1, ax_kplus1, az_kplus1)=self.pqr_ax_az_interpolate(t_approx+dT)
        qdot_k=(q_kplus1-q_k_1)/(2*dT)
        
        return qdot_k

    def backward_diff_q_dot(self, dT, t_approx):
        
        (p_k_1, q_k_1, r_k_1, ax_k_1, az_k_1)=self.pqr_ax_az_interpolate(t_approx-dT)
        (p_k, q_k, r_k, ax_k, az_k)=self.pqr_ax_az_interpolate(t_approx)
        qdot_k=(q_k-q_k_1)/(dT)
        
        return qdot_k

class OnlineNN(nn.Module):	    
            
    def __init__(self, n_hidden_br1=5, n_hidden_br2=5, n_inputs_br1=5, n_inputs_br2=5, n_outputs_br1=2, n_outputs_br2=1, nonlin_br1=torch.tanh, nonlin_br2=torch.tanh, second_layer_br1=False, second_layer_br2=False, Elevator_signal_enabled=True, Thrust_on=False, n_hidden_thrust=5, n_inputs_thrust=2, n_outputs_thrust=2, nonlin_thrust=torch.tanh, second_layer_thrust=False):
        
        super(OnlineNN, self).__init__()

        self.n_inputs_br1=n_inputs_br1
        self.Elevator_signal_enabled=Elevator_signal_enabled
        
        self.n_hidden_br1=n_hidden_br1
        self.n_outputs_br1=n_outputs_br1
        self.second_layer_br1=second_layer_br1 
        
        self.dense0_br1=nn.Linear(self.n_inputs_br1, self.n_hidden_br1)
        torch.nn.init.xavier_uniform_(self.dense0_br1.weight)
        
        if self.second_layer_br1:
            self.dense1_br1=nn.Linear(self.n_hidden_br1, self.n_hidden_br1)
            torch.nn.init.xavier_uniform_(self.dense1_br1.weight)
            
        self.output_br1=nn.Linear(self.n_hidden_br1, self.n_outputs_br1)
        self.nonlin_br1=nonlin_br1
       
    #--------------------------------------------------------------------------
        if self.Elevator_signal_enabled:
            self.n_inputs_br2=n_inputs_br2
        else:
            self.n_inputs_br2=n_inputs_br2-1
            
        self.n_hidden_br2=n_hidden_br2
        self.n_outputs_br2=n_outputs_br2
        self.second_layer_br2=second_layer_br2 
        
        self.dense0_br2=nn.Linear(self.n_inputs_br2, self.n_hidden_br2)
        torch.nn.init.xavier_uniform_(self.dense0_br2.weight)
        
        if self.second_layer_br2:
            self.dense1_br2=nn.Linear(self.n_hidden_br2, self.n_hidden_br2)
            torch.nn.init.xavier_uniform_(self.dense1_br2.weight)
            
        self.output_br2=nn.Linear(self.n_hidden_br2, self.n_outputs_br2)
        self.nonlin_br2=nonlin_br2

    #---------------------------------------------------------------------------
        self.Thrust_on=Thrust_on

        self.n_inputs_thrust=n_inputs_thrust
        self.n_hidden_thrust=n_hidden_thrust
        self.n_outputs_thrust=n_outputs_thrust
        self.second_layer_thrust=second_layer_thrust
	
        self.dense0_thrust=nn.Linear(self.n_inputs_thrust, self.n_hidden_thrust)
        torch.nn.init.xavier_uniform_(self.dense0_thrust.weight)

        if self.second_layer_thrust:
            self.dense1_thrust=nn.Linear(self.n_hidden_thrust, self.n_hidden_thrust)
            torch.nn.init.xavier_uniform_(self.dense1_thrust.weight)

        self.output_thrust=nn.Linear(self.n_hidden_thrust, self.n_outputs_thrust)
        self.nonlin_thrust=nonlin_thrust	
		
    #----------------------------------------------------------------------------- 
	
        self.criterion_ax_az=nn.MSELoss()
        self.criterion_q_dot=q_dot_loss()

	self.validation_set=torch.tensor(np.load('validation_set.npy'), dtype=torch.float32)
	self.validation_set_label=torch.tensor(np.load('validation_set_label.npy'), dtype=torch.float32)
        
    def forward(self, x):

        if self.Thrust_on==False:
       	 
            o1=self.nonlin_br1(self.dense0_br1(x[:, :self.n_inputs_br1]))	
            if self.second_layer_br1:
            		o1=self.nonlin_br1(self.dense1_br1(o1))
            o1=self.output_br1(o1)         
            o2=self.nonlin_br2(self.dense0_br2(x[:, :self.n_inputs_br2]))
            if self.second_layer_br2:
            		o2=self.nonlin_br2(self.dense1_br2(o2))
            o2=self.output_br2(o2)      
            return o1, o2
        else:

            ot=self.nonlin_thrust(self.dense0_thrust(x))
            if self.second_layer_thrust:   	
            		ot=self.nonlin_thrust(self.dense1_thrust(ot))
            ot=self.output_thrust(ot)
            return ot

def training_for_one_sample_period(model, data_buffer, optimizer, scheduler, learning_rate, writer, global_iteration, t_k, t_k_1):
	
    model.train()
				
    while time.time()-t_k<=0.9*data_buffer.Ts:
        
        optimizer.zero_grad()
        (training_set, training_label)=data_buffer.prepare_training_set(model, t_k, t_k_1)
        (out_accel, out_q_dot)=model(training_set)
        
        loss_accel=model.criterion_ax_az(out_accel, training_label[:, :2])
        loss_q_dot=model.criterion_q_dot(label=training_label[:, 2:], estim=out_q_dot, writer=writer, global_iteration=global_iteration)
	
        loss=loss_accel+loss_q_dot
        loss.backward()
        
        scheduler.step(loss)
        optimizer.step()
        
        global_iteration=global_iteration+1
        writer.add_scalar('loss_accel', loss_accel, global_iteration)
        writer.add_scalar('loss_q_dot', loss_q_dot, global_iteration)

    for g in optimizer.param_groups:
        g['lr']=learning_rate
        
    return global_iteration;
    
def estimation_logging(model, data_buffer, t_k, t_k_1, global_iteration, k):
    
    dT=(t_k-t_k_1)/(1+data_buffer.f)    
    data_buffer.processing=True
    (u_b_k, v_b_k, w_b_k, V_k, alpha_k, phi_k, theta_k, ro_k)=data_buffer.prepare_training_sample(t_k)
    (u_b_k_1, v_b_k_1, w_b_k_1, V_k_1, alpha_k_1, phi_k_1, theta_k_1, ro_k_1)=data_buffer.prepare_training_sample(t_k-dT)
                
    (p_k, q_k, r_k, ax_k, az_k)=data_buffer.pqr_ax_az_interpolate(t_k)
    (p_k_1, q_k_1, r_k_1, ax_k_1, az_k_1)=data_buffer.pqr_ax_az_interpolate(t_k-dT)
        
    if model.Elevator_signal_enabled:
        deflection_k=data_buffer.elevator_interpolate(t_k)
        deflection_k_1=data_buffer.elevator_interpolate(t_k_1)
    data_buffer.processing=False

    input_k_array=np.zeros((1, model.n_inputs_br2))
    input_k_1_array=np.zeros((1, model.n_inputs_br2))

    if model.Elevator_signal_enabled:
	input_k_array[0, 0]=alpha_k
	input_k_1_array[0, 0]=alpha_k_1
	input_k_array[0, 1]=q_k
	input_k_1_array[0, 1]=q_k_1
	input_k_array[0, 2]=V_k
	input_k_1_array[0, 2]=V_k_1
	input_k_array[0, 3]=deflection_k
	input_k_1_array[0, 3]=deflection_k_1
    else:
	input_k_array[0, 0]=alpha_k
	input_k_1_array[0, 0]=alpha_k_1
	input_k_array[0, 1]=q_k
	input_k_1_array[0, 1]=q_k_1
	input_k_array[0, 2]=V_k
	input_k_1_array[0, 2]=V_k_1
	
    input_k=torch.tensor(input_k_array, dtype=torch.float32)
    input_k_1=torch.tensor(input_k_1_array, dtype=torch.float32)

    (out_accel_k, out_q_dot_k)=model(input_k)
    (out_accel_k_1, out_q_dot_k_1)=model(input_k_1)
	
    (out_accel_val, out_q_dot_val)=model(model.validation_set)
	
    ax_estimated=out_accel_k.data.tolist()[0][0]
    az_estimated=out_accel_k.data.tolist()[0][1]
    q_estimated=q_k_1+dT*out_q_dot_k_1.data.tolist()[0][0]    
    ax_true=ax_k
    az_true=az_k
    q_true=q_k
    validation_MSE=model.criterion_ax_az(out_accel_val, model.validation_set_label[:, :2])  
    return [ax_estimated, az_estimated, q_estimated, ax_true, az_true, q_true, validation_MSE]    

def visualisation(model, data_buffer, validation_array):
    
    ax_array_from_msgs=np.zeros((len(data_buffer.pqr_ax_az), 1))
    az_array_from_msgs=np.zeros((len(data_buffer.pqr_ax_az), 1))
    q_array_from_msgs=np.zeros((len(data_buffer.pqr_ax_az), 1))
    time_pqr_ax_az=np.zeros((len(data_buffer.pqr_ax_az), 1))

    for i in range(len(data_buffer.pqr_ax_az)):
	time_pqr_ax_az[i, 0]=(data_buffer.pqr_ax_az[i].header.stamp).to_sec()-data_buffer.t_0	
	ax_array_from_msgs[i, 0]=data_buffer.pqr_ax_az[i].linear_acceleration.x
	az_array_from_msgs[i, 0]=data_buffer.pqr_ax_az[i].linear_acceleration.z
	q_array_from_msgs[i, 0]=data_buffer.pqr_ax_az[i].angular_velocity.y

    ax_array_estimated=np.zeros((len(validation_array), 1))
    az_array_estimated=np.zeros((len(validation_array), 1))
    q_array_estimated=np.zeros((len(validation_array), 1))

    ax_array_label=np.zeros((len(validation_array), 1))
    az_array_label=np.zeros((len(validation_array), 1))
    q_array_label=np.zeros((len(validation_array), 1))
	
    validation_MSE=np.zeros((len(validation_array), 1))

    time_pqr_ax_az_k=np.zeros((len(validation_array), 1))

    for i in range(len(validation_array)):
	time_pqr_ax_az_k[i, 0]=data_buffer.time[i]-data_buffer.t_0

	ax_array_estimated[i, 0]=validation_array[i][0]
	az_array_estimated[i, 0]=validation_array[i][1]
	q_array_estimated[i, 0]=validation_array[i][2]

	ax_array_label[i, 0]=validation_array[i][3]
	az_array_label[i, 0]=validation_array[i][4]
	q_array_label[i, 0]=validation_array[i][5]
	
	validation_MSE[i, 0]=validation_array[i][6]
    
    name=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    plt.figure(1)
    plt.plot(time_pqr_ax_az[:, 0], ax_array_from_msgs[:, 0], "b-", time_pqr_ax_az_k[:, 0], ax_array_estimated[:, 0], "r-")
    plt.grid(which='both', axis='both')
    plt.xlabel('time[s]')
    plt.ylabel('ax')
    name1='ax_'+name+'.png'
    plt.savefig(name1)

    plt.figure(2)
    plt.plot(time_pqr_ax_az[:, 0], az_array_from_msgs[:, 0], "b-", time_pqr_ax_az_k[:, 0], az_array_estimated[:, 0], "r-")
    plt.grid(which='both', axis='both')
    plt.xlabel('time[s]')
    plt.ylabel('ay')
    name2='ay_'+name+'.png'
    plt.savefig(name2)

    plt.figure(3)
    plt.plot(time_pqr_ax_az[:, 0], q_array_from_msgs[:, 0], "b-", time_pqr_ax_az_k[:, 0], q_array_estimated[:, 0], "r-")
    plt.grid(which='both', axis='both')
    plt.xlabel('time[s]')
    plt.ylabel('q')
    name3='q_'+name+'.png'
    plt.savefig(name3)
    
    plt.figure(4)
    plt.plot(np.arange(len(validation_array), 0, -1), validation_MSE[:, 0], "b-")
    plt.grid(which='both', axis='both')
    plt.xlabel('k')
    plt.ylabel('Validation MSE')
    name4='MSE_val_k_'+name+'.png'
    plt.savefig(name4) 

    plt.figure(5)
    plt.plot(time_pqr_ax_az_k[:, 0], validation_MSE[:, 0], "b-")
    plt.grid(which='both', axis='both')
    plt.xlabel('time [s]')
    plt.ylabel('Validation MSE')
    name5='MSE_val_time_'+name+'.png'
    plt.savefig(name5) 

    plt.show()   
    
class q_dot_loss(nn.Module):

    def __init__(self):
        super(q_dot_loss, self).__init__()

    def forward(self, label, estim, writer, global_iteration):
        N=label.size()[0]
	
        loss=torch.tensor(0, dtype=torch.float32)
	part_1=torch.tensor(0, dtype=torch.float32)
	part_2=torch.tensor(0, dtype=torch.float32)
        for j in range(N):
	    part_1=part_1+(label[j, 0]-estim)**2
	    part_2=part_2+(label[j, 1]-label[j, 2]-label[j, 3]*estim)**2
	loss=part_1+part_2
	loss=torch.sum(loss/N)

	writer.add_scalar('part_1', torch.sum(part_1), global_iteration)
	writer.add_scalar('part_2', torch.sum(part_2), global_iteration)
        return loss
	
if __name__ == '__main__':
    
    rospy.init_node('Give_this_NN_a_try')
    
    try:
        
        NN = OnlineNN(n_hidden_br1=10, n_hidden_br2=10, n_inputs_br1=3, n_inputs_br2=4, n_outputs_br1=2, n_outputs_br2=1, nonlin_br1=torch.tanh, nonlin_br2=torch.tanh, second_layer_br1=True, second_layer_br2=True, Elevator_signal_enabled=False, Thrust_on=False, n_hidden_thrust=5, n_inputs_thrust=2, n_outputs_thrust=2, nonlin_thrust=torch.tanh, second_layer_thrust=False)

        Buffer=DataBuffer(batch_size=8, frequency=5, updatetime=0.15)
        
        learning_rate=0.001*2*1.3*10

        optimizer=optim.Adam(NN.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.9, 1,verbose=False, threshold=0.0005)
        writer=SummaryWriter()

	validation_array=deque(maxlen=60000)
         
        t_k_1=rospy.get_time()

        m=5
        S=1.5
        global_iteration=0
              
        k=0
	
        while not rospy.is_shutdown():
	            
            if NN.Thrust_on==False and len(Buffer.wind_vel)>=Buffer.train_batch_size \
            and len(Buffer.total_vel)>=Buffer.train_batch_size and len(Buffer.pose)>=Buffer.train_batch_size \
            and len(Buffer.static_p)>=Buffer.train_batch_size and len(Buffer.diff_p)>=Buffer.train_batch_size \
            and len(Buffer.temp)>=Buffer.train_batch_size and len(Buffer.pqr_ax_az)>=Buffer.train_batch_size \
            and len(Buffer.odom)>=Buffer.train_batch_size:
                
                k=k+1                                
                t_k=rospy.get_time()
                Buffer.time.appendleft(t_k)
                global_iteration=training_for_one_sample_period(NN, Buffer, optimizer, scheduler, learning_rate, writer, global_iteration, t_k, t_k_1)
                validation_array.appendleft(estimation_logging(NN, Buffer, t_k, t_k_1, global_iteration, k))		           
                t_k_1=t_k

                print('Executed in:', rospy.get_time()-t_k)
    		print('-'*10)

            elif NN.Thrust_on and len(Buffer.PWM)>=Buffer.train_batch_size:    	    
                
                k=k+1
                t_k=rospy.get_time()
                Buffer.time.appendleft(t_k)

            else:	    
                print('Processing not started')
                rospy.sleep(0.02)
        
        layout={'Acceler':{'ax':['Multiline', ['ax_true', 'ax_estimated']], 'az':['Multiline', ['az_true', 'az_estimated']]}, 'q_dot_central_diff':{'q':['Multiline', ['q_true', 'q_estimated']]}}
        writer.add_custom_scalars(layout)
	visualisation(NN, Buffer, validation_array)


    except rospy.ROSInterruptException:  pass
