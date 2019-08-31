# Copyright 2017    Shihui Yin    Arizona State University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Description: Split large matrix multiplications into small-sized ones and introduce quantization errors for the ADC
# Created on 03/16/2017

import lasagne
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rng = RandomStreams(seed = 12345)

def nonlinear_quant(x, levels, edges):
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

def dec2binary(x, n, n_frac):
    '''
    convert decimal numbers to n-bit unsigned binary numbers
    '''
    y = []
    if n == 1:
        y.append(x)
        return y
    n_int = n - n_frac
    base = 2. ** (n_int - 1)
    x_remain = x
    for i in range(n):
        temp = T.cast(T.ge(x_remain, base), theano.config.floatX)
        y.append(temp)
        x_remain = x_remain - temp * base
        base = base / 2.
    return y

def dec2binary_signed(x, n, n_frac):
    '''
    convert decimal numbers to n-bit signed binary numbers
    '''
    y = []
    if n == 1:
        y.append(x)
        return y
    x_int = x * ((2. ** n_frac) - 1)
    x_unsigned = T.switch(T.ge(x_int, 0.), x_int, (2.**n) + x_int) + 0.5
    base = 2. ** (n-1)
    for i in range(n):
        temp = T.cast(T.ge(x_unsigned, base), theano.config.floatX)
        y.append(temp)
        x_unsigned = x_unsigned - temp * base
        base = base / 2.
    return y

def dec2binary_xnor(x, n, n_frac):
    '''
    convert decimal numbers to n-bit +1/-1 binary numbers
    '''
    y = []
    bound = 2. ** (n - n_frac - 1)
    base = bound * (2. ** (n_frac+1)) / (2. ** (n_frac + 1) - 1.)
    x_remain = x + bound / (2. ** (n_frac + 1) - 1.)
    for i in range(n):
        temp = T.cast(T.ge(x_remain, -bound + base), theano.config.floatX)
        y.append(2. * temp - 1.)
        x_remain = x_remain - temp * base
        base = base / 2
    return y
        
class DenseLayerMonteCarlo_v4(lasagne.layers.DenseLayer):
    '''
    static ADC offsets plus dynamic Gaussian noise, mutliple-bit unsigned input (or 1 bit 
    -1/+1, or 1/0)
    '''
    def __init__(self, incoming, num_units, quant_edges, quant_levels, input_dim=1024, sram_depth=64, 
        bitline_noise=0., ADC_offset=0., n_bits=1, n_frac=0, repr='binary', **kwargs):
        
        self.num_srams = int(np.ceil(np.float32(input_dim) / sram_depth))
        self.sram_depth = sram_depth
        self.input_dim = input_dim
        self.num_edges = quant_edges.shape[0]
        # generate quantization edges for each column (static ADC offsets)
        edges_all_columns = np.expand_dims(np.expand_dims(quant_edges, 1).repeat(num_units, axis=1), 0).repeat(self.num_srams, axis=0)
        if ADC_offset > 0:
            edges_all_columns += np.random.normal(scale=ADC_offset, size=edges_all_columns.shape)
        self.edges_all_columns = theano.shared(edges_all_columns, name='quant_edges', borrow=False)
        self.quant_levels = theano.shared(quant_levels, name='quant_levels', borrow=False)
        self.bitline_noise = bitline_noise
        self.n_bits = n_bits
        self.n_frac = n_frac
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        super(DenseLayerMonteCarlo_v4, self).__init__(incoming, num_units, **kwargs)
        self.activation_list = []
        if repr == 'binary':
            self.dec2binary = dec2binary
            self.base = 2. ** (self.n_bits -self.n_frac - 1)
        if repr == 'xnor':
            self.dec2binary = dec2binary_xnor
            self.base = 2. ** (self.n_bits - 1) / (2. ** (self.n_frac + 1) - 1.)
        if repr == 'twos':
            self.dec2binary = dec2binary_signed
            self.base = 2. ** (self.n_bits - 1) / (2 ** self.n_frac - 1.) if self.n_bits > 1 else 1.
        self.repr = repr   
    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        if self.sram_depth == 1: # no quantization
            activation = T.dot(input, self.W)
            if self.b is not None:
                activation += self.b.dimshuffle('x', 0)
            return activation
        input_list = self.dec2binary(input, self.n_bits, self.n_frac)
        base = self.base
        activation = 0
        for n in range(self.n_bits):
            if n == 0 and self.repr == 'twos':
                base = -self.base
            self.activation_list = []
            for i in range(self.num_srams):
                start_index = i*self.sram_depth
                stop_index = np.min(((i+1)*self.sram_depth, self.input_dim))
                partial_sum = T.dot(input_list[n][:,start_index:stop_index], self.W[start_index:stop_index,:])
                partial_sum_mapped = quantization_by_column_2d(partial_sum, self.edges_all_columns[i], self.quant_levels, self.num_edges, noise=self.bitline_noise, srng=self._srng)
                self.activation_list.append(partial_sum_mapped)
            if self.num_srams > 1:
                activation += T.sum(T.stacklists(self.activation_list), axis=0) * base
            else:
                activation += self.activation_list[0] * base
            if n == 0 and self.repr == 'twos':
                base = -base
            base /= 2.
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
        
class Conv2DLayerMonteCarlo_v4(lasagne.layers.Conv2DLayer):
    '''
    static ADC offsets plus dynamic Gaussian noise
    '''
    def __init__(self, incoming, num_filters, filter_size, quant_edges, quant_levels, 
                 stride=(1, 1), pad=0, untie_biases=False, repr='binary',
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 convolution=T.nnet.conv2d, num_input_channels=128, sram_depth=64,
                 bitline_noise=0., ADC_offset=0., n_bits=1, n_frac=0, **kwargs):
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.sram_depth = sram_depth
        self.convolution = convolution
        self.filter_size = filter_size
        self.n_bits = n_bits
        self.n_frac = n_frac
        self.num_srams = int(np.ceil((self.num_input_channels * self.filter_size[0] * self.filter_size[1] / float(self.sram_depth))))
        self.num_edges = quant_edges.shape[0]
        # generate quantization edges for each column (static ADC offsets)
        edges_all_columns = np.expand_dims(np.expand_dims(quant_edges, 1).repeat(num_filters, axis=1), 0).repeat(self.num_srams, axis=0)
        print(edges_all_columns.shape)
        if ADC_offset > 0:
            edges_all_columns += np.random.normal(scale=ADC_offset, size=edges_all_columns.shape)
        self.edges_all_columns = theano.shared(edges_all_columns, name='quant_edges', borrow=False)
        self.quant_levels = theano.shared(quant_levels, name='quant_levels', borrow=False)
        self.bitline_noise = bitline_noise
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        super(Conv2DLayerMonteCarlo_v4, self).__init__(incoming, num_filters, filter_size, stride=stride,
                                          pad=pad, untie_biases=untie_biases, W=W, b=b,
                                          nonlinearity=nonlinearity, flip_filters=flip_filters,
                                          convolution=convolution)
                                          
        self.partial_input_shape = (self.input_shape[0], self.sram_depth, self.input_shape[2], self.input_shape[3])
        self.W_shape = (self.num_filters, self.sram_depth, self.filter_size[0], self.filter_size[1])
        self.activation_list = []
        if repr == 'binary':
            self.dec2binary = dec2binary
            self.base = 2. ** (self.n_bits -self.n_frac - 1)
        if repr == 'xnor':
            self.dec2binary = dec2binary_xnor
            self.base = 2. ** (self.n_bits - 1) / (2. ** (self.n_frac + 1) - 1.)
        if repr == 'twos':
            self.dec2binary = dec2binary_signed
            self.base = 2. ** (self.n_bits - 1) / (2 ** self.n_frac - 1.) if self.n_bits > 1 else 1.
        self.repr = repr
    
    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad         
        if self.sram_depth == 1:
            return self.convolution(input, 
                                    self.W, 
                                    self.input_shape, self.W_shape,
                                    subsample=self.stride,
                                    border_mode=border_mode,
                                    filter_flip=self.flip_filters)

        activation = 0
        input_list = self.dec2binary(input, self.n_bits, self.n_frac)
        base = self.base
        for n in range(self.n_bits):
            if n == 0 and self.repr == 'twos':
                base = -base
            self.activation_list = []
            num_srams = 0
            if self.sram_depth > self.num_input_channels: # assume num of sram_depth is an integer multiple of num_input_channels
                num_input_channels_per_sram_depth = int(self.sram_depth / self.num_input_channels)
                num_filter_elements = self.filter_size[0] * self.filter_size[1]
                IX = []
                IY = list(range(self.filter_size[1])) * self.filter_size[0]
                for i in range(self.filter_size[0]):
                    IX.extend([i] * self.filter_size[1])
                start_index = 0
                while start_index < num_filter_elements:
                    W_mask = np.zeros((self.num_filters, self.num_input_channels, self.filter_size[0], self.filter_size[1]), dtype=theano.config.floatX)
                    end_index = min(start_index + num_input_channels_per_sram_depth, num_filter_elements)
                    W_mask[:,:,IX[start_index:end_index],IY[start_index:end_index]] = 1
                    
                    start_index += num_input_channels_per_sram_depth
                    
                    partial_input = input_list[n]
                    
                    # Get partial sum
                    partial_sum = self.convolution(partial_input, 
                                            self.W * W_mask, 
                                            self.partial_input_shape, self.W_shape,
                                            subsample=self.stride,
                                            border_mode=border_mode,
                                            filter_flip=self.flip_filters)
                    
                    partial_sum_mapped = quantization_by_column_4d(partial_sum, self.edges_all_columns[num_srams], self.quant_levels, 
                        self.num_edges, noise=self.bitline_noise, srng=self._srng)
                    # Add to the activation_list
                    self.activation_list.append(partial_sum_mapped)
                    num_srams += 1
                
            else:
                num_input_folds = np.int(self.num_input_channels / self.sram_depth) # assume num of input channels is an integer multiple of sram_depth
                for i in range(self.filter_size[0]):
                    for j in range(self.filter_size[1]):
                        W_mask = np.zeros((self.num_filters, self.num_input_channels, self.filter_size[0], self.filter_size[1]), dtype=theano.config.floatX)
                        W_mask[:,:,i,j] = 1
                        for k in range(num_input_folds):
                            input_index = range(k*self.sram_depth,(k+1)*self.sram_depth)
                            partial_input = input_list[n][:,input_index,:,:]
                            
                            partial_sum = self.convolution(partial_input, 
                                                    self.W[:,input_index,:,:] * W_mask[:,input_index,:,:], 
                                                    self.partial_input_shape, self.W_shape,
                                                    subsample=self.stride,
                                                    border_mode=border_mode,
                                                    filter_flip=self.flip_filters)
                            partial_sum_mapped = quantization_by_column_4d(partial_sum, self.edges_all_columns[num_srams], self.quant_levels, 
                                self.num_edges, noise=self.bitline_noise, srng=self._srng)
                            
                            # Add to the activation_list
                            self.activation_list.append(partial_sum_mapped)
                            num_srams += 1
            if (len(self.activation_list) > 1):
                activation += T.sum(T.stacklists(self.activation_list), axis=0) * base
            else:    
                activation += self.activation_list[0] * base
            if n == 0 and self.repr == 'twos':
                base = -base
            base /= 2.
        return activation
  
