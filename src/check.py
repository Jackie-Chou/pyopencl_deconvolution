"""
    check the correctness of customized deconv using tensorflow
    NOTE: only supports no padding now since tensorflow only has SAME and VALID padding type
"""
import tensorflow as tf
import numpy as np
import argparse
import os

class customFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
parser = argparse.ArgumentParser(formatter_class=customFormatter)
parser = argparse.ArgumentParser(formatter_class=customFormatter)
parser.add_argument("-i", "--input_file", type=str, required=True, help="input numpy file (.npz) containing the input tensor & kernel tensor"
                                                                       "\nNOTE the input tensor should be a numpy ndarray of shape (h, w, c)"
                                                                       "\nand dtype np.float32, its name should be 'feature'"
                                                                       "\nthe kernel tensor should be a ndarray of shape (out_c, in_c, kh, kw)"
                                                                       "\nand dtype np.float32, its name should be 'kernel'"
                                                                       "\nNOTE in_c must EQUALS to c")
parser.add_argument("-o", "--output_file", type=str, required=True, help="output numpy file (.npz) containing the output tensor from customized deconv"
                                                                        "\nNOTE the output tensor are ndarray of shape (out_h, out_w, out_c)"
                                                                        "\nand dtype np.float32, its name is 'output'")
parser.add_argument("-hp", "--hpadding", type=int, default=0, help="padding in height dimension")
parser.add_argument("-wp", "--wpadding", type=int, default=0, help="padding in width dimension")
parser.add_argument("-hs", "--hstride", type=int, default=1, help="stride in height dimension")
parser.add_argument("-ws", "--wstride", type=int, default=1, help="stride in width dimension")
parser.add_argument("-hdr", "--hdilation_rate", type=int, default=1, help="dilation rate in height dimension")
parser.add_argument("-wdr", "--wdilation_rate", type=int, default=1, help="dilation rate in width dimension")
parser.add_argument("-g", "--gpu", type=int, required=True, help="gpu id to run on")

if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with np.load(args.input_file) as data:
        feature = data['feature']
        kernel = data['kernel']
    with np.load(args.output_file) as data:
        output = data['output']

    oc, ic, kh, kw = kernel.shape
    f = tf.placeholder(tf.float32, [1] + list(feature.shape))
    k = tf.placeholder(tf.float32, [kh, kw, oc, ic])
    g = tf.placeholder(tf.float32, [1] + list(output.shape))
    o = tf.nn.conv2d_transpose(f, k, [1] + list(output.shape), strides=[1, args.hstride, args.wstride, 1], padding="VALID")
    error = tf.reduce_mean(tf.squared_difference(g, o))

    with tf.Session(config=tf.ConfigProto(log_device_placement=False, \
                    gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        er = sess.run(error, feed_dict={f: np.expand_dims(feature, 0),
                                        k: np.transpose(kernel, [2, 3, 0, 1]),
                                        g: np.expand_dims(output, 0)})

    print("average squared error: {}".format(er))



