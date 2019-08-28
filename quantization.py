
import lasagne
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rng = RandomStreams(seed = 12345)

def nonlinear_quant(x, levels, edges):
    # Perform fast ideal nonlinear quantization based on given quantization edges and levels
    if len(levels) == 4:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], 
            T.switch(T.lt(x, edges[2]), levels[2], levels[3])))
    if len(levels) == 5:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], 
            T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], levels[4]))))
    if len(levels) == 6:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], 
            T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], levels[5])))))
    if len(levels) == 7:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], 
            T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5], levels[6]))))))
    if len(levels) == 8:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], 
            T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5], T.switch(T.lt(x, edges[6]), levels[6], levels[7])))))))
    if len(levels) == 9:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], 
            T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5], T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], levels[8]))))))))
    if len(levels) == 10:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4],
            T.switch(T.lt(x, edges[5]), levels[5], T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], levels[9])))))))))
    if len(levels) == 11:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4],
            T.switch(T.lt(x, edges[5]), levels[5], T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], T.switch(T.lt(x, edges[9]), levels[9], levels[10]))))))))))
    if len(levels) == 12:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5],
            T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], T.switch(T.lt(x, edges[9]), levels[9], T.switch(T.lt(x, edges[10]), levels[10], levels[11])))))))))))
    if len(levels) == 13:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5],
            T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], T.switch(T.lt(x, edges[9]), levels[9], T.switch(T.lt(x, edges[10]), levels[10], T.switch(T.lt(x, edges[11]), levels[11], levels[12]))))))))))))
    if len(levels) == 14:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5],
            T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], T.switch(T.lt(x, edges[9]), levels[9], T.switch(T.lt(x, edges[10]), levels[10], T.switch(T.lt(x, edges[11]), levels[11], T.switch(T.lt(x, edges[12]), levels[12], levels[13])))))))))))))
    if len(levels) == 15:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5],
            T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], T.switch(T.lt(x, edges[9]), levels[9], T.switch(T.lt(x, edges[10]), levels[10], T.switch(T.lt(x, edges[11]), levels[11], T.switch(T.lt(x, edges[12]), levels[12], T.switch(T.lt(x, edges[13]), levels[13], levels[14]))))))))))))))
    if len(levels) == 16:
        y = T.switch(T.lt(x, edges[0]), levels[0], T.switch(T.lt(x, edges[1]), levels[1], T.switch(T.lt(x, edges[2]), levels[2], T.switch(T.lt(x, edges[3]), levels[3], T.switch(T.lt(x, edges[4]), levels[4], T.switch(T.lt(x, edges[5]), levels[5],
            T.switch(T.lt(x, edges[6]), levels[6], T.switch(T.lt(x, edges[7]), levels[7], T.switch(T.lt(x, edges[8]), levels[8], T.switch(T.lt(x, edges[9]), levels[9], T.switch(T.lt(x, edges[10]), levels[10], T.switch(T.lt(x, edges[11]), levels[11], T.switch(T.lt(x, edges[12]), levels[12], T.switch(T.lt(x, edges[13]), levels[13], T.switch(T.lt(x, edges[14]), levels[14], levels[15])))))))))))))))
    return y
    
def quantization_by_column_4d(x, edges_all_columns, levels, num_edges, noise=0, srng=None):
    '''
    Apply column-specific quantization to x, before quantization, dynamic noise is added optionally
    x: a 4-d tensor to be quantized (batch_size, columns, map_x, map_y)
    edges_all_columns: a 2-d matrix, quantization edges for all the columns (num_edges, columns)
    levels: a 1-d vector, quantization levels shared by all the columns
    num_edges: number of quantization edges (number of levels - 1)
    noise: sigma of Gaussian noise added to x before quantization
    srng: random number generator for additive Gaussian noise
    '''
    if noise > 0 and srng is not None:
        x = x + srng.normal(x.shape, avg=0.0, std=noise)
    sum = 0
    for i in range(num_edges):
        edge = edges_all_columns[i,:].dimshuffle('x', 0, 'x', 'x')
        sum += T.cast(T.gt(x, edge), dtype='int32')
    return levels[sum]
    
def quantization_by_column_2d(x, edges_all_columns, levels, num_edges, noise=0, srng=None):
    '''
    Apply column-specific quantization to x, before quantization, dynamic noise is added optionally
    x: a 2-d tensor to be quantized (batch_size, columns)
    edges_all_columns: a 2-d matrix, quantization edges for all the columns (num_edges, columns)
    levels: a 1-d vector, quantization levels shared by all the columns
    num_edges: number of quantization edges (number of levels - 1)
    noise: sigma of Gaussian noise added to x before quantization
    srng: random number generator for additive Gaussian noise
    '''
    if noise > 0 and srng is not None:
        x = x + srng.normal(x.shape, avg=0.0, std=noise)
    sum = 0
    for i in range(num_edges):
        edge = edges_all_columns[i,:].dimshuffle('x', 0)
        sum += T.cast(T.gt(x, edge), dtype='int32')
    return levels[sum]

