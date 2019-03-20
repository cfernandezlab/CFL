from .network import Network
import tensorflow as tf

class LayoutEstimator_EquiConvs(Network):
    def setup(self):
        feed_dict_test = {}
        feed_dict_train = {}
        self.nname = "edge-estimator"
        with tf.variable_scope(self.nname):
            (self.feed('rgb_input')
                 .equi_conv(7, 7, 64, 2, 2, 1, relu=False, name='conv1')
                 .batch_normalization(relu=True, name='bn_conv1')
                 .max_pool(3, 3, 2, 2, name='pool1')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                 .batch_normalization(name='bn2a_branch1'))

            (self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                 .batch_normalization(relu=True, name='bn2a_branch2a')
                 .equi_conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2a_branch2b')
                 .batch_normalization(relu=True, name='bn2a_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                 .batch_normalization(name='bn2a_branch2c'))

            (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
                 .add(name='res2a')
                 .relu(name='res2a_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
                 .batch_normalization(relu=True, name='bn2b_branch2a')
                 .equi_conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2b_branch2b')
                 .batch_normalization(relu=True, name='bn2b_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
                 .batch_normalization(name='bn2b_branch2c'))

            (self.feed('res2a_relu', 
                   'bn2b_branch2c')
                 .add(name='res2b')
                 .relu(name='res2b_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
                 .batch_normalization(relu=True, name='bn2c_branch2a')
                 .equi_conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2c_branch2b')
                 .batch_normalization(relu=True, name='bn2c_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
                 .batch_normalization(name='bn2c_branch2c'))

            (self.feed('res2b_relu', 
                   'bn2c_branch2c')
                 .add(name='res2c')
                 .relu(name='res2c_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
                 .batch_normalization(name='bn3a_branch1'))

            (self.feed('res2c_relu')
                 .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
                 .batch_normalization(relu=True, name='bn3a_branch2a')
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3a_branch2b')
                 .batch_normalization(relu=True, name='bn3a_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
                 .batch_normalization(name='bn3a_branch2c'))
    
            (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
                 .add(name='res3a')
                 .relu(name='res3a_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
                 .batch_normalization(relu=True, name='bn3b_branch2a')
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3b_branch2b')
                 .batch_normalization(relu=True, name='bn3b_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
                 .batch_normalization(name='bn3b_branch2c'))

            (self.feed('res3a_relu', 
                   'bn3b_branch2c')
                 .add(name='res3b')
                 .relu(name='res3b_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
                 .batch_normalization(relu=True, name='bn3c_branch2a')
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3c_branch2b')
                 .batch_normalization(relu=True, name='bn3c_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
                 .batch_normalization(name='bn3c_branch2c'))

            (self.feed('res3b_relu', 
                   'bn3c_branch2c')
                 .add(name='res3c')
                 .relu(name='res3c_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
                 .batch_normalization(relu=True, name='bn3d_branch2a')
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3d_branch2b')
                 .batch_normalization(relu=True, name='bn3d_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
                 .batch_normalization(name='bn3d_branch2c'))

            (self.feed('res3c_relu', 
                   'bn3d_branch2c')
                 .add(name='res3d')
                 .relu(name='res3d_relu')
                 .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1')
                 .batch_normalization(name='bn4a_branch1'))


            (self.feed('res3d_relu')
                 .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
                 .batch_normalization(relu=True, name='bn4a_branch2a')
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4a_branch2b')
                 .batch_normalization(relu=True, name='bn4a_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
                 .batch_normalization(name='bn4a_branch2c'))


            (self.feed('bn4a_branch1', 
                   'bn4a_branch2c')
                 .add(name='res4a')
                 .relu(name='res4a_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b_branch2a')
                 .batch_normalization(relu=True, name='bn4b_branch2a')
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4b_branch2b')
                 .batch_normalization(relu=True, name='bn4b_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b_branch2c')
                 .batch_normalization(name='bn4b_branch2c'))

            (self.feed('res4a_relu', 
                   'bn4b_branch2c')
                 .add(name='res4b')
                 .relu(name='res4b_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4c_branch2a')
                 .batch_normalization(relu=True, name='bn4c_branch2a')
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4c_branch2b')
                 .batch_normalization(relu=True, name='bn4c_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4c_branch2c')
                 .batch_normalization(name='bn4c_branch2c'))

            (self.feed('res4b_relu', 
                   'bn4c_branch2c')
                 .add(name='res4c')
                 .relu(name='res4c_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4d_branch2a')
                 .batch_normalization(relu=True, name='bn4d_branch2a')
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4d_branch2b')
                 .batch_normalization(relu=True, name='bn4d_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4d_branch2c')
                 .batch_normalization(name='bn4d_branch2c'))

            (self.feed('res4c_relu', 
                   'bn4d_branch2c')
                 .add(name='res4d')
                 .relu(name='res4d_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4e_branch2a')
                 .batch_normalization(relu=True, name='bn4e_branch2a')
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4e_branch2b')
                 .batch_normalization(relu=True, name='bn4e_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4e_branch2c')
                 .batch_normalization(name='bn4e_branch2c'))

            (self.feed('res4d_relu', 
                   'bn4e_branch2c')
                 .add(name='res4e')
                 .relu(name='res4e_relu')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4f_branch2a')
                 .batch_normalization(relu=True, name='bn4f_branch2a')
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4f_branch2b')
                 .batch_normalization(relu=True, name='bn4f_branch2b')
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4f_branch2c')
                 .batch_normalization(name='bn4f_branch2c'))

            (self.feed('res4e_relu', 
                   'bn4f_branch2c')
                 .add(name='res4f')
                 .relu(name='res4f_relu')
                 .conv(1, 1, 2048, 2, 2, biased=False, relu=False, name='res5a_branch1')
                 .batch_normalization(name='bn5a_branch1'))

            (self.feed('res4f_relu')
                 .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res5a_branch2a')
                 .batch_normalization(relu=True, name='bn5a_branch2a')
                 .equi_conv(3, 3, 512, 1, 1, 1, biased=False, relu=False, name='res5a_branch2b')
                 .batch_normalization(relu=True, name='bn5a_branch2b')
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
                 .batch_normalization(name='bn5a_branch2c'))

            (self.feed('bn5a_branch1', 
                   'bn5a_branch2c')
                 .add(name='res5a')
                 .relu(name='res5a_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
                 .batch_normalization(relu=True, name='bn5b_branch2a')
                 .equi_conv(3, 3, 512, 1, 1, 1, biased=False, relu=False, name='res5b_branch2b')
                 .batch_normalization(relu=True, name='bn5b_branch2b')
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
                 .batch_normalization(name='bn5b_branch2c'))
            
            drop_out_d = tf.placeholder(tf.float32,name = "drop_out_d")
            feed_dict_train[drop_out_d] = 0.5 
            feed_dict_test[drop_out_d] = 1.0
                        
            (self.feed('res5a_relu', 
                   'bn5b_branch2c') 
                 .add(name='res5b')
                 .relu(name='res5b_relu')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
                 .batch_normalization(relu=True, name='bn5c_branch2a',dropout=drop_out_d)
                 .equi_conv(3, 3, 512, 1, 1, 1, biased=False, relu=False, name='res5c_branch2b')
                 .batch_normalization(relu=True, name='bn5c_branch2b',dropout=drop_out_d)
                 .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
                 .batch_normalization(name='bn5c_branch2c'))
                
            #------------------------------------------------------------------------------------  
            # decoder EDGE MAPS & CORNERS MAPS
            (self.feed('bn5c_branch2c') 
                 .equi_conv(3, 3, 512, 1, 1, 1, biased=True, relu=True, name='d_2x_ec')
                 .bilinear_unpool(2, name='d_2x'))
            (self.feed('d_2x','res4f_relu')
                 .concat(axis=3,name="d_concat_2x")
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=True, relu=True, name='d_4x_ec')
                 .bilinear_unpool(2, name='d_4x')
                 .equi_conv(3, 3, 2, 1, 1, 1, biased=True, relu=False, name='output4X_likelihood'))
            (self.feed('d_4x','res3d_relu','output4X_likelihood')
                 .concat(axis=3,name="d_concat_4x")
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=True, relu=True, name='d_8x_ec')
                 .bilinear_unpool(2, name='d_8x')
                 .equi_conv(3, 3, 2, 1, 1, 1, biased=True, relu=False, name='output8X_likelihood'))
            (self.feed('d_8x','res2c_relu','output8X_likelihood')
                 .concat(axis=3,name="d_concat_8x")
                 .equi_conv(5, 5, 64, 1, 1, 1, biased=True, relu=True, name='d_16x_ec')
                 .bilinear_unpool(2, name='d_16x')
                .equi_conv(3, 3, 2, 1, 1, 1, biased=True, relu=False, name='output16X_likelihood'))
            (self.feed('d_16x','bn_conv1','output16X_likelihood')
                 .concat(axis=3,name="d_concat_16x")
                 .equi_conv(5, 5, 64, 1, 1, 1, biased=True, relu=True, name='d_16x_conv1')
                 .equi_conv(3, 3, 2, 1, 1, 1, biased=True, relu=False, name='output_likelihood'))
               
                 
        self.fd_test = feed_dict_test
        self.fd_train = feed_dict_train