#!/usr/bin/env bash
set -e
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
NSYNC_INC=$TF_INC"/external/nsync/public"
# please modify $ARCH according to the following list and your gpu model.
ARCH=sm_52


# If coming across: cudaCheckError() failed : invalid device function. change -arch=sm_xx accordingly.

# Which CUDA capabilities do we want to pre-build for?
# https://developer.nvidia.com/cuda-gpus
#   Compute/shader model   Cards
#   6.1		      P4, P40, Titan X so CUDA_MODEL = 61
#   6.0                    P100 so CUDA_MODEL = 60
#   5.2                    M40
#   3.7                    K80
#   3.5                    K40, K20
#   3.0                    K10, Grid K520 (AWS G2)
#   Other Nvidia shader models should work, but they will require extra startup
#   time as the code is pre-optimized for them.
# CUDA_MODELS=30 35 37 52 60 61

if [ "$TF_INC" = "" ]; then
  echo "Tensorflow include dir is empty"
  return "1"
fi



CUDA_HOME=/usr/local/cuda-10.0/

echo "Configuration variables:"
echo "Tensorflow Include directory: $TF_INC" 
echo "Tensorflow Library directory: $TF_LIB" 
echo "Nvidia Arch: $ARCH"
read -p "Do you want to continue with the compilation with this configuration (Y/n)?" choice

case "$choice" in 
		[yY][eE][sS]|y|Y|"" ) CONTINUE="1";;
		[nN][oO]|n|N ) CONTINUE="0";;
		* ) echo "invalid";;
esac

if [ "$CONTINUE" = "1" ]; then

  #if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
  #cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/


  cd deform_conv_layer
  nvcc -std=c++11 -ccbin=/usr/bin/g++-4.9 -c -o deform_conv.cu.o deform_conv.cu.cc -I $TF_INC -I $NSYNC_INC -D\
            GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L $CUDA_HOME/lib64 --expt-relaxed-constexpr -arch=$ARCH -DNDEBUG


  ## if you install tf using already-built binary, or gcc version 4.x, uncomment the three lines below


  g++-4.9 -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o -I\
        $TF_INC -I $NSYNC_INC -fPIC -lcudart -L $CUDA_HOME/lib64 -D GOOGLE_CUDA=1 -Wfatal-errors \
        -L $TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0 
  # for gcc5-built tf
  # g++ -std=c++11 -shared -o deform_conv.so deform_conv.cc deform_conv.cu.o \
  #   -I $TF_INC -I $NSYNC_INC -fPIC -D GOOGLE_CUDA -lcudart -L $CUDA_HOME/lib64 -L $TF_LIB -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI=0

  cd ..
else
  echo "Modify the make.sh file to math your configuration."
fi
