import numpy as np
import tensorflow as tf
import re
import math
from config import * 
from .deform_conv_layer import deform_conv_op as deform_conv_op


DEFAULT_PADDING = 'SAME'
DEFAULT_TYPE = tf.float32


def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

summary = True
def ActivationSummary(layer): #tensorBoard (jmfacil)
    if summary:
        TOWER_NAME = 'tower'
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', layer.op.name)
        tf.summary.histogram(tensor_name + '/activations', layer)   


@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, is_training = True,bs=16):#,reuse=None): #cfernandez
        self.inputs = []
        self.batch_size = bs
        self.layers = dict(inputs)
        self.trainable = trainable
        self.is_training = is_training
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        def transform_names(k):
            if k == 'mean':
                return 'moving_mean'
            if k == 'variance':
                return 'moving_variance'
            if k == 'scale':
                return 'gamma'
            if k == 'offset':
                return 'beta'
            return k

        print(data_path)
        data_dict = np.load(data_path,encoding='latin1').item()
        for key in data_dict:
            superkey=self.nname+"/"+key
            with tf.variable_scope(superkey, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        nsubkey=transform_names(subkey)
                        var = tf.get_variable(nsubkey)                        
                        session.run(var.assign(data_dict[key][subkey]))
                    except ValueError:
                        print("ignore "+key,subkey)
                        print(superkey,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=superkey))
                        if not ignore_missing:

                            raise
        print ("Loaded weitghts")                        

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_layer_output(self, name):
        return self.layers[name]    

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in list(self.layers.items())) + 1
        return '%s_%d' % (prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    def filler(self, params): #chema
        #print "Filler: "+str(params)
        value = params.get("value",0.0)
        mean = params.get("mean",0.0)
        std = params.get("std",0.1)
        dtype = params.get("dtype",DEFAULT_TYPE)
        name = params.get("name",None)
        uniform = params.get("uniform",False)
        return {
                "xavier_conv2d" : tf.contrib.layers.xavier_initializer_conv2d(uniform = uniform),
                "t_normal" : tf.truncated_normal_initializer(mean = mean, stddev = std, dtype = dtype) ,
                "constant" : tf.constant_initializer(value = value, dtype = dtype)
                }[params.get("type","t_normal")]


    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, rate=1, biased=True, relu=True, padding=DEFAULT_PADDING, trainable=True, initializer=None):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.convolution(
            i, k, padding=padding, strides=[s_h, s_w], dilation_rate=[rate, rate])
        with tf.variable_scope(name,reuse=False) as scope: #cfernandez reuse

            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            #kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
             #                      regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // 1, c_o],initializer=self.filler({ "type" : "t_normal", #cfernandez
                                                                                            "mean" : 0.0,
                                                                                            "std"  : 0.1
                                                                                            }),regularizer=self.l2_regularizer(args.weight_decay)) #0.0005 cfg.TRAIN.WEIGHT_DECAY

            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    output = tf.nn.relu(bias)
                output = tf.nn.bias_add(conv, biases)    
 
            else:
                conv = convolve(input, kernel)
                if relu:
                    output = tf.nn.relu(conv)
                output = conv    
            
            return output

    @staticmethod
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
    
    @staticmethod
    def equi_coord(pano_W,pano_H,k_W,k_H,u,v): 
        """ contribution by cfernandez and jmfacil """
        fov_w = k_W * np.deg2rad(360./float(pano_W))
        focal = (float(k_W)/2) / np.tan(fov_w/2)
        c_x = 0
        c_y = 0

        u_r, v_r = u, v 
        u_r, v_r = u_r-float(pano_W)/2.,v_r-float(pano_H)/2.
        phi, theta = u_r/(pano_W) * (np.pi) *2, -v_r/(pano_H) * (np.pi)

        ROT = Network.rotation_matrix((0,1,0),phi)
        ROT = np.matmul(ROT,Network.rotation_matrix((1,0,0),theta))#np.eye(3)

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
        
        phi = np.arctan2(rays[0,...],rays[2,...])
        theta = np.arcsin(np.clip(rays[1,...],-1,1))
        x = (pano_W)/(2.*np.pi)*phi +float(pano_W)/2.
        y = (pano_H)/(np.pi)*theta +float(pano_H)/2.
        
        roi_y = h_grid+v_r +float(pano_H)/2.
        roi_x = w_grid+u_r +float(pano_W)/2.

        new_roi_y = (y) 
        new_roi_x = (x) 

        offsets_x = (new_roi_x - roi_x)
        offsets_y = (new_roi_y - roi_y)

        return offsets_x, offsets_y

    @staticmethod
    def equi_coord_fixed_resoltuion(pano_W,pano_H,k_W,k_H,u,v,pano_Hf = -1, pano_Wf=-1): 
        """ contribution by cfernandez and jmfacil """
        pano_Hf = pano_H if pano_Hf<=0 else pano_H/pano_Hf
        pano_Wf = pano_W if pano_Wf<=0 else pano_W/pano_Wf
        fov_w = k_W * np.deg2rad(360./float(pano_Wf))
        focal = (float(k_W)/2) / np.tan(fov_w/2)
        c_x = 0
        c_y = 0

        u_r, v_r = u, v 
        u_r, v_r = u_r-float(pano_W)/2.,v_r-float(pano_H)/2.
        phi, theta = u_r/(pano_W) * (np.pi) *2, -v_r/(pano_H) * (np.pi)

        ROT = Network.rotation_matrix((0,1,0),phi)
        ROT = np.matmul(ROT,Network.rotation_matrix((1,0,0),theta))#np.eye(3)

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
        
        phi = np.arctan2(rays[0,...],rays[2,...])
        theta = np.arcsin(np.clip(rays[1,...],-1,1))
        x = (pano_W)/(2.*np.pi)*phi +float(pano_W)/2.
        y = (pano_H)/(np.pi)*theta +float(pano_H)/2.
        
        roi_y = h_grid+v_r +float(pano_H)/2.
        roi_x = w_grid+u_r +float(pano_W)/2.

        new_roi_y = (y) 
        new_roi_x = (x) 

        offsets_x = (new_roi_x - roi_x)
        offsets_y = (new_roi_y - roi_y)

        return offsets_x, offsets_y

    @staticmethod
    def distortion_aware_map(pano_W, pano_H, k_W, k_H, s_width = 1, s_height = 1,bs = 16):
        """ contribution by cfernandez and jmfacil """
        n=1
        offset = np.zeros(shape=[pano_H,pano_W,k_H*k_W*2])
        print(offset.shape)
        
        for v in range(0, pano_H, s_height): 
            for u in range(0, pano_W, s_width): 
                offsets_x, offsets_y = Network.equi_coord_fixed_resoltuion(pano_W,pano_H,k_W,k_H,u,v,1,1)
                offsets = np.concatenate((np.expand_dims(offsets_y,-1),np.expand_dims(offsets_x,-1)),axis=-1)
                total_offsets = offsets.flatten().astype("float32")
                offset[v,u,:] = total_offsets
                
        offset = tf.constant(offset)
        offset = tf.expand_dims(offset, 0)
        offset = tf.concat([offset for _ in range(bs)],axis=0)
        offset = tf.cast(offset, tf.float32)
            
        return offset
    

    @layer
    def equi_conv(self, input, k_h, k_w, c_o, s_h, s_w, num_deform_group, name, num_groups = 1, rate = 1, biased=True, relu=True, 
                    padding=DEFAULT_PADDING, trainable=True, initializer=None):
        """ contribution by cfernandez and jmfacil """
        self.validate_padding(padding)
        data = input
        n,h,w,_ = tuple(data.get_shape().as_list())
        data_shape = data.shape
        offset = tf.stop_gradient(Network.distortion_aware_map(w, h, k_w, k_h, s_width = s_w, s_height = s_h,bs= self.batch_size))

        c_i = data.get_shape()[-1]
        trans2NCHW = lambda x:tf.transpose(x, [0, 3 ,1 ,2])
        trans2NHWC = lambda x:tf.transpose(x, [0, 2 ,3, 1])
        # deform conv only supports NCHW
        data = trans2NCHW(data)
        offset = trans2NCHW(offset)
        dconvolve = lambda i, k, o: deform_conv_op.deform_conv_op(
            i, k, o, strides = [1, 1, s_h, s_w], rates=[1, 1, rate, rate], padding=padding, num_groups=num_groups, deformable_group=num_deform_group)
        with tf.variable_scope(name, reuse=False) as scope:

            init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(args.weight_decay))
            kernel = tf.transpose(kernel,[3,2,0,1])
            ActivationSummary(offset)

            print(data, kernel, offset)
            dconv = trans2NHWC(dconvolve(data, kernel, offset))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(dconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(dconv, biases)
            else:
                if relu:
                    return tf.nn.relu(dconv)
                return dconv
      

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride=2, name='upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True, initializer=None):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape_d = tf.shape(input)
        in_shape = input.shape.as_list()
        if shape is None:
            h = ((in_shape[1]) * stride)
            w = ((in_shape[2]) * stride)
            new_shape = [in_shape_d[0], h, w, c_o]
        else:
            new_shape = [in_shape_d[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name,reuse=False) as scope:
            init_weights = tf.zeros_initializer() if initializer is 'zeros' else tf.contrib.layers.variance_scaling_initializer(
                factor=0.01, mode='FAN_AVG', uniform=False) #cfernandez
            filters = self.make_var('weights', filter_shape, init_weights, trainable,
                                   regularizer=self.l2_regularizer(args.weight_decay)) #cfg.TRAIN.WEIGHT_DECAY
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)


            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    output = tf.nn.relu(bias)
                output = tf.nn.bias_add(deconv, biases)  

            else:
                if relu:
                    output = tf.nn.relu(deconv)
                output = devonv    
            return output    


    @layer
    def reduce_max(self,input_data, name):
        return tf.reduce_max(input_data, axis=1, keep_dims=True)

    @layer
    def reduce_mean(self,input_data, name):
        return tf.reduce_mean(input_data, axis=1, keep_dims=True)

    @layer
    def argmax(self,input_data, name):
        return tf.argmax(input_data, axis=1)

    @layer
    def bilinear_unpool(self,input_data,mul_factor, name):
        _,h,w,_ = tuple(input_data.get_shape().as_list())
        return tf.image.resize_bilinear(input_data,(h*mul_factor,w*mul_factor),align_corners=True,name=name)

    @layer
    def mul_grad(self,input_data, mul,name):
        return (1.0-mul)*tf.stop_gradient(input_data)+(mul)*input_data    


    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print(input)
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def psroi_pool(self, input, output_dim, group_size, spatial_scale, name):
        """contribution by miraclebiu"""
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        return psroi_pooling_op.psroi_pool(input[0], input[1],
                                           output_dim=output_dim,
                                           group_size=group_size,
                                           spatial_scale=spatial_scale,
                                           name=name)[0]


    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)
    @layer
    def reshape(self, input, shape, name):
        return tf.reshape(input, shape=shape, name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)
    @layer
    def flatten_data(self, input, name):
        return tf.reshape(input, shape=[input.shape[0], -1], name=name)
    
    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    # The original
    @layer
    def batch_normalization(self,input,name,relu=True, dropout=None): #, is_training= True): #, is_training= True
        #jmfacil/cfernandez: dropout added based on pix2pix
        is_training =  self.is_training
        #is_training=False
        if dropout is not None:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            if relu:
                temp_layer = tf.nn.relu(temp_layer)
            #output = tf.nn.dropout(temp_layer,dropout)
            return tf.nn.dropout(temp_layer,dropout)
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            #output = tf.nn.relu(temp_layer)
            return tf.nn.relu(temp_layer)
        else:
            #output = tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

        #ActivationSummary(output)
        #return output    

    @layer
    def batch_normalization0(self,input,name,relu=True, is_training=True, dropout=None, scale_offset=True, decay = 0.999):
        shape = [input.get_shape()[-1]]
        with tf.variable_scope(name,reuse=False) as scope:
            if scale_offset:
                scale = self.make_var('gamma', shape=shape,
                            initializer=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            )
                        )
                offset = self.make_var('beta', shape=shape)
            else:
                scale, offset = (None, None)
        
            pop_mean = self.make_var('moving_mean', shape=shape)
            pop_var = self.make_var('moving_variance', shape=shape,
                            initializer=self.filler(
                                {"type":"constant",
                                 "value": 1.0}
                            ),
                            regularizer = False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(input, [0,1,2], name='moments')
            train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                epsilon=1e-4
                output =  tf.nn.batch_normalization(input,
                    batch_mean, batch_var, offset, scale, epsilon)
        else:
            epsilon=1e-4
            output = tf.nn.batch_normalization(input,pop_mean, pop_var, offset, scale, epsilon)
        #jmfacil/cfernandez: dropout added based on pix2pix
        if dropout is not None:
            #temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            #if relu:
            #    temp_layer = tf.nn.relu(temp_layer)
            output = tf.nn.dropout(output,dropout)
        """contribution by miraclebiu"""
        if relu:
            #temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            output = tf.nn.relu(output)
        #else:
            #return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
        return output    
    


    @layer
    def scale(self, input, c_in, name):
        with tf.variable_scope(name,reuse=False) as scope:

            alpha = tf.get_variable('alpha', shape=[c_in, ], dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0), trainable=True,
                                    regularizer=self.l2_regularizer(0.00001))
            beta = tf.get_variable('beta', shape=[c_in, ], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0), trainable=True,
                                   regularizer=self.l2_regularizer(0.00001))
            return tf.add(tf.multiply(input, alpha), beta)


    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


