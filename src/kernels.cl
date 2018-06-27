__kernel void deconv(__global const float *fvec, 
                     __global const float *kvec, 
                     __global float *ovec,
                     const int flen)
{
    int oc, ncols;
    int c, j, i;
    int base;
    float result;

    ncols = get_global_size(0);
    oc = get_global_size(1);
    j = get_global_id(0);
    c = get_global_id(1), 
    base = c*(ncols*flen) + j*flen;
        
    result = 0.0f;
    for(i = 0; i < flen; ++ i)
        result += fvec[i] * kvec[base + i];

    ovec[c*ncols+j] = result;
}
