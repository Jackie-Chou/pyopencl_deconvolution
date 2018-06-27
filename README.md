# pyopencl_deconvolution
a simple deconvolution implemented with **pyopencl**

complete configuration involving paddings, strides, dilation rates in both spatial dimensions is supported

the basic idea is using matrix multiplication to implement deconvolution

## file description

check: folder contains opencl code to check the configuration of the machine and opencl

src: folder contains main source code

- deconv.py: main code of deconvolution, NOTE the platform choice is hard coded to 0, one may feel free to change it. run *python deconv.py -h* for arguments and usage.also NOTE this file reads input feature map and kernel from input .npz file, and result will be stored to an output .npz file, example files please refer to *input.npz* and *output.npz*

- kernels.cl: opencl kernel function file,  NOTE the implementation here is to let one work-item process a whole element in output tensor, which may not be optimal yet.

- check.py: a simple correctness checking script by compare the result with what tensorflow outputs, average squared difference error is used to measure the difference. NOTE *deconv.py* do support a complete configuration involving paddings, strides, dilation rates in both spatial dimensions, while tensorflow do not(at least in tf.nn.conv2d_transpose), so correctness checking should only be done when the configuration is compatible with tensorflow, and also feel free to change the tensorflow code if you want.

- pipeline.py: a simple pipeline script, it will automatically generate a input .npz file *input.npz* and run *deconv.py*

  