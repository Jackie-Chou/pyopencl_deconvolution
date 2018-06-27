import pyopencl as cl
import pyopencl.array
import numpy as np
import argparse

class customFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
parser = argparse.ArgumentParser(formatter_class=customFormatter)
parser.add_argument("-i", "--input_file", type=str, required=True, help="input numpy file (.npz) containing the input tensor & kernel tensor"
                                                                       "\nNOTE the input tensor should be a numpy ndarray of shape (h, w, c)"
                                                                       "\nand dtype np.float32, its name should be 'feature'"
                                                                       "\nthe kernel tensor should be a ndarray of shape (out_c, in_c, kh, kw)"
                                                                       "\nand dtype np.float32, its name should be 'kernel'"
                                                                       "\nNOTE in_c must EQUALS to c")
parser.add_argument("-o", "--output_file", type=str, required=True, help="output numpy file (.npz) containing the output tensor"
                                                                        "\nNOTE the output tensor are ndarray of shape (out_h, out_w, out_c)"
                                                                        "\nand dtype np.float32, its name is 'output'")
parser.add_argument("-hp", "--hpadding", type=int, default=0, help="padding in height dimension")
parser.add_argument("-wp", "--wpadding", type=int, default=0, help="padding in width dimension")
parser.add_argument("-hs", "--hstride", type=int, default=1, help="stride in height dimension")
parser.add_argument("-ws", "--wstride", type=int, default=1, help="stride in width dimension")
parser.add_argument("-hdr", "--hdilation_rate", type=int, default=1, help="dilation rate in height dimension")
parser.add_argument("-wdr", "--wdilation_rate", type=int, default=1, help="dilation rate in width dimension")
parser.add_argument("-g", "--gpu", type=int, required=True, help="gpu id to run on")

def parse_input(input_file):
    with np.load(input_file) as data:
        feature = data["feature"]
        kernel = data['kernel']
    
    return feature, kernel

'''
def transform_input(feature, kernel, args):
    """
        transform the raw feature and kernel under given configuration(padding, dilation_rate)
        into equivalent features and kernels with canonical configuration(padding=0, dilation=1)
    """
    h, w, c = feature.shape
    oc, ic, kh, kw = kernel.shape
    new_h = h + 2 * args.hpadding
    new_w = w + 2 * args.wpadding
    new_kh = (kh-1) * args.hdilation_rate + 1
    new_kw = (kw-1) * args.wdilation_rate + 1

    new_feature = np.zeros((new_h, hew_w, c), dtype=np.float32)
    new_kernel = np.zeros((oc, ic, new_kh, new_kw), dtype=np.float32)

    new_feature[hpadding:-hpadding, wpadding:-wpadding] = feature
    hdr = args.hdilation_rate
    hwr = args.wdilation_rate
    for i in range(kh):
        for j in range(kw):
            new_kernel[i*hdr, j*hwr] = kernel[i, j]

    return new_feature, new_kernel
'''

def unroll(feature, kernel, args):
    """
        unroll the tensors into vectors to perform convolution(deconvolution) efficiently
    """
    h, w, c = feature.shape
    oc, ic, kh, kw = kernel.shape
    hp, wp = args.hpadding, args.wpadding
    hs, ws = args.hstride, args.wstride
    hdr, wdr = args.hdilation_rate, args.wdilation_rate

    # dilated kernel size
    new_kh = (kh-1) * hdr + 1
    new_kw = (kw-1) * wdr + 1

    # compute output shape(with padding)
    oh = (h - 1)*hs + new_kh
    ow = (w - 1)*ws + new_kw
    
    # enroll feature tensor into vector
    fvec = np.ndarray(shape=(h*w*c,), dtype=np.float32)
    for k in range(c):
        for i in range(h):
            fvec[k*(h*w)+i*w: k*(h*w)+(i+1)*w] = feature[i, :, k]

    # enroll kernel into vector
    # first dilate the kernel for facility
    kvec = np.ndarray(shape=(oc*h*w*oh*ow*ic,), dtype=np.float32)
    for ci in range(oc):
        for cj in range(ic):
            # convert to matrix
            one_vector = np.zeros([oh*ow], dtype=np.float32)
            for i in range(kh):
                for j in range(kw):
                    one_vector[(i*hdr)*ow+j*wdr] = kernel[ci, cj, i, j]

            one_matrix = np.ndarray(shape=[oh*ow, h*w], dtype=np.float32)
            for j in range(h*w):
                one_matrix[:, j] = np.roll(one_vector, (j/w)*hs*ow + (j%w)*ws)
            one_matrix = one_matrix.T
    
            # store to long-vector
            for j in range(oh*ow):
                base = ci*h*w*ic*oh*ow+j*h*w*ic+cj*h*w 
                kvec[base: base+h*w] = one_matrix[:, j]

    return fvec, kvec, [oh, ow, oc]

def roll(ovec, oh, ow, oc):
    """
        convert the long output vector into tensor of shape (oh, ow, oc)
    """
    output = np.ndarray(shape=(oh, ow, oc), dtype=np.float32)
    for k in range(oc):
        for i in range(oh):
            output[i, :, k] = ovec[k*oh*ow+i*ow: k*oh*ow+(i+1)*ow]

    return output

def main():
    args = parser.parse_args()
    platform = cl.get_platforms()[0]
    print platform

    device = platform.get_devices()[args.gpu]
    print device

    context = cl.Context([device])
    print context

    program = cl.Program(context, open("kernels.cl").read()).build()
    print program

    queue = cl.CommandQueue(context)
    print queue

    feature, kernel = parse_input(args.input_file)
    print("feature shape: {}".format(feature.shape))
    print("kernel shape: {}".format(kernel.shape))
    fvec, kvec, output_shape = unroll(feature, kernel, args)
    oh, ow, oc = tuple(output_shape)
    print("output shape: {}".format((oh, ow, oc)))

    mem_flags = cl.mem_flags
    feature_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, \
                           hostbuf=fvec)
    kernel_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, \
                           hostbuf=kvec)
    ovec = np.ndarray(shape=(oh*ow*oc,), dtype=np.float32)
    destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, ovec.nbytes)

    program.deconv(queue, (oh*ow, oc), None, feature_buf, kernel_buf, destination_buf, np.int32(len(fvec)))
    cl.enqueue_copy(queue, ovec, destination_buf)
    output = roll(ovec, *output_shape)

    # cut off padding if not zero
    hp, wp = args.hpadding, args.wpadding
    output = output[hp: oh-hp, wp: ow-wp]
    print("final shape: {}".format(output.shape))

    # store to output file
    np.savez(args.output_file, output=output)

if __name__ == "__main__":
    main()
