
4
keras_deconv_valid_inputPlaceholder*
dtype0
V
keras_deconv_valid/ShapeShapekeras_deconv_valid_input:0*
T0*
out_type0
�
keras_deconv_valid/kernelConst*�
value�B�"�u��>P�7�{�>�>V�l�=v<���c\�i��>Y�>6�������,�=�]K>�Ĕ>���=(�`���Э������t��閾��<���>��>��> r!�.����&1�pm���x�=`~���7>��r=4��=Є�<�t��ý�2�;����f�M�0&�<�ɂ�ͳ�>����#>&Ok>���=;�f����=�N>�5!;짅=@��*X\>�`�@���g>>E+�>�nQ> }���>�����V3^>L�ѽ
�*>�:d��+>�����>H��� #ڽ*
dtype0
T
keras_deconv_valid/biasConst*%
valueB"                *
dtype0
T
&keras_deconv_valid/strided_slice/stackConst*
valueB: *
dtype0
V
(keras_deconv_valid/strided_slice/stack_1Const*
valueB:*
dtype0
V
(keras_deconv_valid/strided_slice/stack_2Const*
valueB:*
dtype0
�
 keras_deconv_valid/strided_sliceStridedSlicekeras_deconv_valid/Shape&keras_deconv_valid/strided_slice/stack(keras_deconv_valid/strided_slice/stack_1(keras_deconv_valid/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0
V
(keras_deconv_valid/strided_slice_1/stackConst*
valueB:*
dtype0
X
*keras_deconv_valid/strided_slice_1/stack_1Const*
valueB:*
dtype0
X
*keras_deconv_valid/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
"keras_deconv_valid/strided_slice_1StridedSlicekeras_deconv_valid/Shape(keras_deconv_valid/strided_slice_1/stack*keras_deconv_valid/strided_slice_1/stack_1*keras_deconv_valid/strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
Index0*
T0
V
(keras_deconv_valid/strided_slice_2/stackConst*
valueB:*
dtype0
X
*keras_deconv_valid/strided_slice_2/stack_1Const*
valueB:*
dtype0
X
*keras_deconv_valid/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
"keras_deconv_valid/strided_slice_2StridedSlicekeras_deconv_valid/Shape(keras_deconv_valid/strided_slice_2/stack*keras_deconv_valid/strided_slice_2/stack_1*keras_deconv_valid/strided_slice_2/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
B
keras_deconv_valid/mul/yConst*
value	B :*
dtype0
d
keras_deconv_valid/mulMul"keras_deconv_valid/strided_slice_1keras_deconv_valid/mul/y*
T0
B
keras_deconv_valid/add/yConst*
value	B :*
dtype0
X
keras_deconv_valid/addAddkeras_deconv_valid/mulkeras_deconv_valid/add/y*
T0
D
keras_deconv_valid/mul_1/yConst*
value	B :*
dtype0
h
keras_deconv_valid/mul_1Mul"keras_deconv_valid/strided_slice_2keras_deconv_valid/mul_1/y*
T0
D
keras_deconv_valid/add_1/yConst*
value	B :*
dtype0
^
keras_deconv_valid/add_1Addkeras_deconv_valid/mul_1keras_deconv_valid/add_1/y*
T0
D
keras_deconv_valid/stack/3Const*
value	B :*
dtype0
�
keras_deconv_valid/stackPack keras_deconv_valid/strided_slicekeras_deconv_valid/addkeras_deconv_valid/add_1keras_deconv_valid/stack/3*
N*
T0*

axis 
�
#keras_deconv_valid/conv2d_transposeConv2DBackpropInputkeras_deconv_valid/stackkeras_deconv_valid/kernelkeras_deconv_valid_input:0*
paddingVALID*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
keras_deconv_valid/BiasAddBiasAdd#keras_deconv_valid/conv2d_transposekeras_deconv_valid/bias*
T0*
data_formatNHWC 