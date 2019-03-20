from .network import Network
import tensorflow as tf

class LayoutEstimator_EquiConvs(Network):
    def setup(self):
        feed_dict_test = {}
        feed_dict_train = {}
        self.nname = "edge-estimator"
        with tf.variable_scope(self.nname):
            (self.feed('rgb_input')
                 #.conv(7, 7, 64, 2, 2, relu=False, name='conv1')#def
                 .equi_conv(7, 7, 64, 2, 2, 1, relu=False, name='conv1')#def
                 .batch_normalization(relu=True, name='bn_conv1')
                 .max_pool(3, 3, 2, 2, name='pool1')#def
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
                 .batch_normalization(name='bn2a_branch1'))

            (self.feed('pool1')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
                 .batch_normalization(relu=True, name='bn2a_branch2a')
                 #.conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')#def
                 .equi_conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2a_branch2b')#def
                 .batch_normalization(relu=True, name='bn2a_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
                 .batch_normalization(name='bn2a_branch2c'))

            (self.feed('bn2a_branch1', 
                   'bn2a_branch2c')
                 .add(name='res2a')
                 .relu(name='res2a_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
                 .batch_normalization(relu=True, name='bn2b_branch2a')
                 #.conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')#def
                 .equi_conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2b_branch2b')#def
                 .batch_normalization(relu=True, name='bn2b_branch2b')
                 .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
                 .batch_normalization(name='bn2b_branch2c'))

            (self.feed('res2a_relu', 
                   'bn2b_branch2c')
                 .add(name='res2b')
                 .relu(name='res2b_relu')
                 .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
                 .batch_normalization(relu=True, name='bn2c_branch2a')
                 #.conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')#def
                 .equi_conv(3, 3, 64, 1, 1, 1, biased=False, relu=False, name='res2c_branch2b')#def
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
                 #.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')#def
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3a_branch2b')#def
                 .batch_normalization(relu=True, name='bn3a_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
                 .batch_normalization(name='bn3a_branch2c'))
    
            (self.feed('bn3a_branch1', 
                   'bn3a_branch2c')
                 .add(name='res3a')
                 .relu(name='res3a_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b_branch2a')
                 .batch_normalization(relu=True, name='bn3b_branch2a')
                 #.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b_branch2b')#def
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3b_branch2b')#def
                 .batch_normalization(relu=True, name='bn3b_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b_branch2c')
                 .batch_normalization(name='bn3b_branch2c'))

            (self.feed('res3a_relu', 
                   'bn3b_branch2c')
                 .add(name='res3b')
                 .relu(name='res3b_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3c_branch2a')
                 .batch_normalization(relu=True, name='bn3c_branch2a')
                 #.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3c_branch2b')#def
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3c_branch2b')#def
                 .batch_normalization(relu=True, name='bn3c_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3c_branch2c')
                 .batch_normalization(name='bn3c_branch2c'))

            (self.feed('res3b_relu', 
                   'bn3c_branch2c')
                 .add(name='res3c')
                 .relu(name='res3c_relu')
                 .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3d_branch2a')
                 .batch_normalization(relu=True, name='bn3d_branch2a')
                 #.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3d_branch2b')#def
                 .equi_conv(3, 3, 128, 1, 1, 1, biased=False, relu=False, name='res3d_branch2b')#def
                 .batch_normalization(relu=True, name='bn3d_branch2b')
                 .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3d_branch2c')
                 .batch_normalization(name='bn3d_branch2c'))

            (self.feed('res3c_relu', 
                   'bn3d_branch2c')
                 .add(name='res3d')
                 .relu(name='res3d_relu')
                 .conv(1, 1, 1024, 2, 2, biased=False, relu=False, name='res4a_branch1')
                 .batch_normalization(name='bn4a_branch1'))

            drop_out_d = tf.placeholder(tf.float32,name = "drop_out_d")
            feed_dict_train[drop_out_d] = 0.5 #0.5 
            feed_dict_test[drop_out_d] = 1.0

            (self.feed('res3d_relu')
                 .conv(1, 1, 256, 2, 2, biased=False, relu=False, name='res4a_branch2a')
                 .batch_normalization(relu=True, name='bn4a_branch2a',dropout=drop_out_d)
                 #.conv(3, 3, 256, 1, 1, biased=False, relu=False, name='res4a_branch2b')#def
                 .equi_conv(3, 3, 256, 1, 1, 1, biased=False, relu=False, name='res4a_branch2b')#def
                 .batch_normalization(relu=True, name='bn4a_branch2b',dropout=drop_out_d)
                 .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
                 .batch_normalization(name='bn4a_branch2c'))
                 
            #------------------------------------------------------------------------------------     
            # decoder EDGEMAPS & CORNERS
            (self.feed('bn4a_branch2c') 
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
                 #.bilinear_unpool(2, name='d_8x')
                 .equi_conv(3, 3, 2, 1, 1, 1, biased=True, relu=False, name='output_likelihood'))

            '''(self.feed('bn4a_branch2c') 
                 .upconv(None,256,ksize=5,stride=2,name='d_4x', biased=True, relu=True)
                 .upconv(None, 2, ksize=3,stride=1,biased=True,relu=False, name = 'output4X_likelihood')) 
            (self.feed('d_4x','res3d_relu','output4X_likelihood')
                 .concat(axis=3,name="d_concat_4x")
                 .upconv(None, 128,ksize=5, stride=2,biased=True, relu=True, name='d_8x')
                 .upconv(None, 2, ksize=3, stride=1,relu=False, biased=True,name = 'output8X_likelihood'))
            (self.feed('d_8x','res2c_relu','output8X_likelihood')
                 .concat(axis=3,name="d_concat_8x")
                 .upconv(None, 64, ksize=5,stride= 2,biased=True, relu=True, name='d_16x'))
            (self.feed('d_16x')
                 .concat(axis=3,name="d_concat_16x")
                 .upconv(None,64, ksize=3,stride= 1, biased=True, relu=True, name='d_16x_conv1')               
                 .upconv(None, 2, ksize=3, stride=1,biased=True,relu=False, name = 'output_likelihood')) '''                 
                 
        self.fd_test = feed_dict_test;
        self.fd_train = feed_dict_train;
        

''' For each step, the code will generate a feed dictionary that will contain the set of examples on which to train for the step, keyed by the placeholder ops they represent.

images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)

A python dictionary object is then generated with the placeholders as keys and the representative feed tensors as values.

feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}
This is passed into the sess.run() function's feed_dict parameter to provide the input examples for this step of training.'''

