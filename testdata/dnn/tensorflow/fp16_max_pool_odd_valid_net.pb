
$
input_15Placeholder*
dtype0
�
conv2d_16/kernelConst*�
value�B�j����]�����R���Z�����`�f�W�����������X�h�����a�h�b�������`�S�j�`�i�������`���c���������M���e�����k�i���I�j�T��*
dtype0
>
conv2d_16/biasConst*
valueBj   *
dtype0
�
conv2d_17/convolutionConv2Dinput_15conv2d_16/kernel*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
T0
c
conv2d_17/BiasAddBiasAddconv2d_17/convolutionconv2d_16/bias*
T0*
data_formatNHWC
�
max_pooling2d_5/MaxPoolMaxPoolconv2d_17/BiasAdd*
strides
*
data_formatNHWC*
ksize
*
paddingVALID*
T0