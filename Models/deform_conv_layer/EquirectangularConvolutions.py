import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt
import math
from matplotlib import colors as mcolors
import tensorflow as tf
import deform_conv_op as deform_conv_op

#import models

def make_var(name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

def l2_regularizer(weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def equi_coord(pano_W,pano_H,k_W,k_H,u,v): 
    fov_w = k_W * np.deg2rad(360./float(pano_W))
    focal = (float(k_W)/2) / np.tan(fov_w/2)
    c_x = 0
    c_y = 0

    #phi, theta = ((u- pano_W/2 -0.5)/pano_W * math.pi *2, -(v- pano_H/2 -0.5)/pano_H * math.pi)
    u_r, v_r = u, v #u+0.5, v+0.5
    u_r, v_r = u_r-float(pano_W)/2.,v_r-float(pano_H)/2.
    phi, theta = u_r/pano_W * np.pi *2, -v_r/pano_H * np.pi
    
    ROT = rotation_matrix((0,1,0),phi)
    ROT = np.matmul(ROT,rotation_matrix((1,0,0),theta))#np.eye(3)

    h_range = np.array(range(k_H))
    w_range = np.array(range(k_W))
    w_ones = (np.ones(k_W))
    h_ones = (np.ones(k_H))
    h_grid = np.matmul(np.expand_dims(h_range,-1),np.expand_dims(w_ones,0))+0.5-float(k_H)/2
    w_grid = np.matmul(np.expand_dims(h_ones,-1),np.expand_dims(w_range,0))+0.5-float(k_W)/2
    
    K=np.array([[focal,0,c_x],[0,focal,c_y],[0.,0.,1.]])
    inv_K = np.linalg.inv(K)
    rays = np.stack([w_grid,h_grid,np.ones(h_grid.shape)],0)
    rays = np.matmul(inv_K,rays.reshape(3,k_H*k_W))
    rays /= np.linalg.norm(rays,axis=0,keepdims=True)
    rays = np.matmul(ROT,rays)
    rays=rays.reshape(3,k_H,k_W)
    
    '''theta = np.arctan2(rays[0,...],rays[2,...])
    phi = np.arcsin(np.clip(rays[1,...],-1,1))
    PI = np.pi
    y = pano_H*(phi + 0.5*PI)/PI 
    x = pano_W*0.5*(theta/PI+1)'''

    phi = np.arctan2(rays[0,...],rays[2,...])
    theta = np.arcsin(np.clip(rays[1,...],-1,1))
    x = (pano_W)/(2.*np.pi)*phi +float(pano_W)/2.
    y = (pano_H)/(np.pi)*theta +float(pano_H)/2. 
    
    '''roi_y = h_grid+v
    roi_x = w_grid+u
    new_roi_y = (y)+0.5
    new_roi_x = (x)+0.5'''

    roi_y = h_grid + v_r +float(pano_H)/2.
    roi_x = w_grid + u_r +float(pano_W)/2.
    new_roi_y = y
    new_roi_x = x

    offsets_x = (new_roi_x - roi_x)
    offsets_y = (new_roi_y - roi_y)

    '''colors = np.random.rand(len(roi_x.flatten()))
    imgplot = plt.imshow(im)
    plt.scatter( roi_x.flatten(), roi_y.flatten(), s=2, c='r', alpha=0.5)
    plt.scatter( new_roi_x.flatten(), new_roi_y.flatten(), s=2, c='b', alpha=0.5)
    plt.show()'''

    return offsets_x, offsets_y

DEFAULT_PADDING = 'SAME'
def equi_conv(input,offset,kernel, k_h, k_w, c_o, s_h, s_w, num_deform_group, name, num_groups = 1, rate = 1, biased=True, relu=True, 
    padding=DEFAULT_PADDING, trainable=True, initializer=None):
    
    """ contribution by miraclebiu, and biased option"""
    #self.validate_padding(padding)
    data = input
    n,h,w,_ = tuple(data.get_shape().as_list())
    
    #n= 1 #cambiaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    #offset = tf.zeros(shape=[1,h,w,k_h*k_w*2], dtype=tf.float32)
    
    c_i = data.get_shape()[-1]
    
    trans2NCHW = lambda x:tf.transpose(x, [0, 3 ,1 ,2])
    trans2NHWC = lambda x:tf.transpose(x, [0, 2 ,3, 1])
    trans2khkwcico = lambda x:tf.transpose(x, [2, 3 ,1 ,0]) # for the kernel
    
    # deform conv only supports NCHW
    data = trans2NCHW(data)
    offset = trans2NCHW(offset)
    dconvolve = lambda i, k, o: deform_conv_op.deform_conv_op(
        i, k, o, strides = [1, 1, s_h, s_w], rates=[1, 1, rate, rate], padding=padding, num_groups=num_groups, deformable_group=num_deform_group)
    
    with tf.variable_scope(name, reuse=False) as scope:

        # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
        init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
            factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        kernel = tf.expand_dims(tf.expand_dims(kernel,-1),-1)
        #kernel = make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
        #                       regularizer=None)
        kernel = tf.transpose(kernel,[3,2,0,1])
        
        print(data, kernel, offset)
        dconv = trans2NHWC(dconvolve(data, kernel, offset))

        if biased:
            biases = make_var('biases', [c_o], init_biases, trainable)
            if relu:
                bias = tf.nn.bias_add(dconv, biases)
                return tf.nn.relu(bias)
            return tf.nn.bias_add(dconv, biases)
        else:
            if relu:
                return tf.nn.relu(dconv)
            return dconv  

'''def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width = 1, s_height = 1):
    n=1
    offset = np.zeros(shape=[pano_H,pano_W,k_H*k_W*2])
    print(offset.shape)
    
    for v in range(0, pano_H, s_height): 
        for u in range(0, pano_W, s_width): 
            new_roi_x, new_roi_y = equi_coord(pano_W,pano_H,k_W,k_H,u,v)
            new_roi_x = new_roi_x
            new_roi_x = new_roi_x.flatten()
            new_roi_y = new_roi_y
            new_roi_y = new_roi_y.flatten()
            #total_roi = np.concatenate((new_roi_x, new_roi_y), axis=None).astype("float32")
            total_roi = []
            for i in range(0,len(new_roi_x)):    
                total_roi = np.concatenate((total_roi,new_roi_y[i], new_roi_x[i]), axis=None).astype("float32")
            
            offset[v,u,:] = total_roi
            
    offset = tf.constant(offset)
    offset = tf.expand_dims(offset, 0)
    offset = tf.cast(offset, tf.float32)
    #offset = tf.shape(tf.expand_dims(offset, 0))
        
    return offset'''

def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width = 1, s_height = 1,bs = 1,**kargs):
        n=1
        offset = np.zeros(shape=[pano_H,pano_W,k_H*k_W*2])
        print(offset.shape)
        
        for v in range(0, pano_H, s_height): 
            for u in range(0, pano_W, s_width): 
                offsets_x, offsets_y = equi_coord(pano_W,pano_H,k_W,k_H,u,v)
                offsets = np.concatenate((np.expand_dims(offsets_y,-1),np.expand_dims(offsets_x,-1)),axis=-1)
                #offsets_x, offsets_y = (offsets_x.T,offsets_y.T)
                #offsets_x = offsets_x.flatten()
                #offsets_y = offsets_y.flatten()
                
                #total_offsets = np.concatenate((offsets_y, offsets_x), axis=None).astype("float32")
                total_offsets = offsets.flatten().astype("float32")
                #total_offsets = []
                #for i in range(0,len(offsets_x)):    
                    #total_offsets = np.concatenate((total_offsets,offsets_x[i], offsets_y[i]), axis=None).astype("float32")

                offset[v,u,:] = total_offsets
                #print("offset",total_roi)
                
        offset = tf.constant(offset)
        offset = tf.expand_dims(offset, 0)
        print('OFFSET BEFORE', offset)
        offset = tf.concat([offset for _ in range(bs)],axis=0)
        print('OFFSET AFTER', offset)
        offset = tf.cast(offset, tf.float32)
        #offset = tf.shape(tf.expand_dims(offset, 0))
            
        return offset    

def distortion_aware_map1(pano_W, pano_H, k_W, k_H, s_width = 1, s_height = 1,traspose=False,xy=0):
    
    n=1
    offset = np.zeros(shape=[pano_H,pano_W,k_H*k_W*2])
    print(offset.shape)
    print(pano_H,pano_W)
    
    for v in range(0, pano_H, s_height): 
        for u in range(0, pano_W, s_width): 
            new_roi_x, new_roi_y = equi_coord(pano_W,pano_H,k_W,k_H,u,v)
            new_roi_x, new_roi_y = (new_roi_x.T,new_roi_y.T) if traspose else (new_roi_x, new_roi_y)
            new_roi_x = new_roi_x.flatten()
            new_roi_y = new_roi_y.flatten()
            if xy==0:
                total_roi = np.concatenate((new_roi_y, new_roi_x), axis=None).astype("float32")
            elif xy==1:
                total_roi = np.concatenate((new_roi_x, new_roi_y), axis=None).astype("float32")
            elif xy==2:    
                total_roi = []
                for i in range(0,len(new_roi_x)):    
                    total_roi = np.concatenate((total_roi,new_roi_y[i], new_roi_x[i]), axis=None).astype("float32")
            else:
                total_roi = []
                for i in range(0,len(new_roi_x)):    
                    total_roi = np.concatenate((total_roi,new_roi_x[i], new_roi_y[i]), axis=None).astype("float32")
            offset[v,u,:] = total_roi
            
    offset = tf.convert_to_tensor(offset)
    offset = tf.expand_dims(offset, 0)
    #offset = tf.shape(tf.expand_dims(offset, 0))
        
    return offset    

# MAIN
#def main():

# Image RGB
im = Image.open("/misc/lab106_d2/datasets/Sun360/Images/pano1024x512/indoor/childs_room_daycare/pano_aclzqydjlssfry.jpg").convert("L")
im = im.resize((512,256), resample=Image.LANCZOS)
pano_W, pano_H = im.size
images = []
images.append(np.array(im))
#images.append(np.array(im))
images = np.expand_dims(np.asarray(images), axis=3)
print(images.shape)

# Create a placeholder for the input image
channels = 1
input_pano = tf.placeholder(tf.float32, shape=(None, pano_H, pano_W, channels))
print(input_pano)

# 3x3 sobel filter
#sobel_x = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32) #Laplacian
filt = np.zeros((1,61))
filt[0,0]=1
#sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
sobel_x = tf.constant(filt, tf.float32)
k_H, k_W = tuple(sobel_x.shape.as_list())
sobel_x_filter = tf.reshape(sobel_x, [k_H, k_W, 1, 1])
#kernel = np.array(np.random.rand(3,3))
#k_H, k_W = kernel.shape 

# Equirectangular offsets
offset = distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width = 1, s_height = 1,traspose=True,xy=0)
offset = tf.cast(offset, tf.float32)
# Sobel filter by standard convolution
filtered_x_std_conv = tf.nn.conv2d(input_pano, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
result_std_conv = tf.abs(filtered_x_std_conv)

# Sobel filter by distortion aware convolution
c_o = 1
s_h = 1 
s_w = 1
num_deform_group = 1
result_dist_conv = equi_conv(input_pano, offset,sobel_x, k_H, k_W, c_o, s_h, s_w, num_deform_group, 'panoR', num_groups = 1, rate = 1, biased=True, relu=True, 
                padding=DEFAULT_PADDING, trainable=True, initializer=None)


'''u, v = (1024,100) #application center in pixels
kernel = np.array(np.random.rand(71,71))
k_H, k_W = kernel.shape 
equi_coord(pano_W,pano_H,k_W,k_H,u,v)'''

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    #plt.imshow(images[0, :, :, 0], cmap = "gray")
    #plt.show()
        
    filtered = sess.run(result_std_conv, feed_dict={input_pano: images})
    print(filtered.shape)

    filtered2 = sess.run(result_dist_conv, feed_dict={input_pano: images})
    print(filtered2.shape)

    '''plt.imshow(filtered[0, :, :, 0], cmap = "gray")
    plt.show()
    plt.imshow(filtered2[0, :, :, 0], cmap = "gray")
    plt.show()'''

    plt.subplot(121)
    plt.imshow(filtered[0, :, :, 0], cmap = "gray")
    plt.subplot(122)
    plt.imshow(filtered2[0, :, :, 0], cmap = "gray")
    plt.show()

    '''plt.subplot(311)
    plt.imshow(filtered[0, :, :, 0], cmap = "gray")
    plt.subplot(312)
    plt.imshow(images[0, :, :, 0], cmap = "gray")
    plt.subplot(313)
    plt.imshow(filtered2[0, :, :, 0], cmap = "gray")
    plt.show()'''

    '''diff = filtered - filtered2
    plt.imshow(diff[0, :, :, 0], cmap = "gray")
    plt.show()'''


'''if __name__ == '__main__':
main()'''