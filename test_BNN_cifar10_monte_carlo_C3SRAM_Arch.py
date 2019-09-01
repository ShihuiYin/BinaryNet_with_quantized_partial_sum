
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

import lasagne
import theano
import theano.tensor as T

import binary_ops
import XNORNET_SRAM

import sys
import scipy.io as sio
import csv
import argparse


from pylearn2.datasets.cifar10 import CIFAR10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNN CIFAR10 mapped on C3SRAM-based architecture", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mc', '--monte_carlo_runs', default=1, type=int, help="Number of monte carlo runs")
    parser.add_argument('-rs', '--result_filename', default='results.mat', type=str, help="Result file name to save all the test errors from Monte Carlo runs")
    parser.add_argument('-qt', '--quant_filename', default='quant_256_11_13.mat', type=str, help="quantization file defining quantization edges and levels")
    parser.add_argument('-sd', '--sram_depth', default=256, type=int, help="SRAM depth of C3SRAM")
    parser.add_argument('-bn', '--bitline_noise', default=0.97, type=float, help="Standard deviation of bitline noise in terms of MAC value")
    parser.add_argument('-ao', '--ADC_offset', default=0.89, type=float, help="ADC offset in terms of MAC value")
    parser.add_argument('-nb', '--n_bits', default=1, type=int, help="Activation precision, currently only support 1-bit in this script")
    parser.add_argument('-fb', '--n_frac', default=0, type=int, help="Number of fraction bits in activation, currently only support 0-bit in this script")
    parser.add_argument('-rp', '--repr', default='xnor', type=str, help="Representation of binary activation: xnor(-1/+1), binary(+1/0), currently only support xnor in this script")
    parser.add_argument('-bs', '--batch_size', default=50, type=int, help="Batch size")
    parser.add_argument('-ni', '--num_images', default=10000, type=int, help="number of images to test")
    parser.add_argument('-ac', '--arch', default='128-256-256-512', type=str, help="Architecture of model: currently support 128-256-512-1024 and 128-256-256-512")
    parser.add_argument('-bl', '--baseline', action='store_true', help="If '--baseline' specified, baseline accuracy will be evaluated.")
    args = parser.parse_args()
    monte_carlo_runs = args.monte_carlo_runs
    result_filename = args.result_filename
    quant_filename = args.quant_filename
    sram_depth = args.sram_depth if not args.baseline else 1
    bitline_noise = args.bitline_noise
    ADC_offset = args.ADC_offset
    n_bits = args.n_bits
    n_frac = args.n_frac
    repr = args.repr
    
    batch_size = args.batch_size
    num_images = args.num_images
    num_batches = int(num_images / batch_size)

    layers = list(map(int,args.arch.split('-')))
    if args.arch == "128-256-256-512":
        model_filename = 'cifar10_parameters_128_256_256_512.npz'
    elif args.arch == "128-256-512-1024":
        model_filename = 'cifar10_parameters_BNN.npz'
    else:
        print("Currently, architecture %s is not supported!" % args.arch)
        exit
    activation = binary_ops.SignTheano
    quant_lower_bound = -sram_depth
    print("batch_size = "+str(batch_size))
    dict = sio.loadmat(quant_filename)
    quant_edges = np.squeeze(dict['edges'].astype('float32'))
    quant_levels = np.squeeze(dict['levels'].astype('float32'))
    
    print('Loading CIFAR-10 dataset...')
    
    test_set = CIFAR10(which_set= 'test')
    test_set.X = np.reshape(np.subtract(np.multiply(2./255., test_set.X), 1.), (-1,3,32,32))[0:num_images]
    
    # flatten targets
    test_set.y = np.hstack(test_set.y)[0:num_images]
    #dict_temp = {'test_y': test_set.y}
    #sio.savemat('CIFAR10_labels.mat', dict_temp)   
    # Onehot the targets
    test_set.y = np.float32(np.eye(10)[test_set.y])

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    
    mlp = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),input_var=input)
    
    # Input layer is not binary -> use baseline kernel in first hidden layer
            
    cnn = lasagne.layers.InputLayer(
                shape=(None, 3, 32, 32),
                input_var=input)            
    
    cnn = lasagne.layers.Conv2DLayer(
                cnn,
                num_filters=layers[0],
                filter_size=(3, 3),
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity)
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.Conv2DLayerMonteCarlo_v4(
                cnn,
                num_filters=layers[0],
                filter_size=(3, 3),
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity,
                num_input_channels=layers[0],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.Conv2DLayerMonteCarlo_v4(
                cnn,
                num_filters=layers[1],
                filter_size=(3, 3),
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity,
                num_input_channels=layers[0],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.Conv2DLayerMonteCarlo_v4(
                cnn,
                num_filters=layers[1],
                filter_size=(3, 3),
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity,
                num_input_channels=layers[1],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.Conv2DLayerMonteCarlo_v4(
                cnn,
                num_filters=layers[2],
                filter_size=(3, 3),
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity,
                num_input_channels=layers[1],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.Conv2DLayerMonteCarlo_v4(
                cnn,
                num_filters=layers[2],
                filter_size=(3, 3),
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                pad=1,
                nonlinearity=lasagne.nonlinearities.identity,
                num_input_channels=layers[2],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.DenseLayerMonteCarlo_v4(
                cnn,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=layers[3],
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                input_dim=layers[2]*16,
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    cnn = XNORNET_SRAM.DenseLayerMonteCarlo_v4(
                cnn,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=layers[3],
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                input_dim=layers[3],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.BatchNormLayer(cnn)
    cnn = lasagne.layers.NonlinearityLayer(cnn,nonlinearity=activation)
    
    
    cnn = XNORNET_SRAM.DenseLayerMonteCarlo_v4(
                cnn,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=10,
                quant_edges=quant_edges,
                quant_levels=quant_levels,
                input_dim=layers[3],
                sram_depth=sram_depth,
                n_bits=n_bits,
                n_frac=n_frac,
                repr=repr,
                bitline_noise=bitline_noise,
                ADC_offset=ADC_offset)
                
    cnn = lasagne.layers.BatchNormLayer(cnn)

    
    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    
    val_fn = theano.function([input, target], [test_err, test_output], on_unused_input='ignore')
    
    #val_fn = theano.function([input, target], input_FC, on_unused_input='ignore')
    print("Loading the trained parameters and binarizing the weights...")
    
    # Load parameters
    with np.load(model_filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)

    
    # Binarize the weights
    params = lasagne.layers.get_all_params(cnn)
    for param in params:
        # print param.name
        if param.name == "W":
            param.set_value(binary_ops.SignNumpy(param.get_value()))
    
    print('Running...')
    
    start_time = time.time()
    test_error_list = []
    test_logits_epochs = []
       
    for j in range(monte_carlo_runs):
        
        test_error = 0
        test_logits_lists = []
        for i in range(num_batches):
            print(" batch%d/%d" %(i+1, num_batches))
            [test_error_mb, test_logits_mb] = val_fn(test_set.X[(i*batch_size):(i+1)*batch_size],test_set.y[(i*batch_size):(i+1)*batch_size])
            test_error += test_error_mb * 100
            test_logits_lists.append(test_logits_mb)
            print("Time elasped:%ss" % (time.time() - start_time))
        print("Run %d:test_error = %.2f%%" % (j, (test_error/num_batches)))
        test_logits_epoch = np.vstack(test_logits_lists)
        test_logits_epochs.append(test_logits_epoch)
        test_error_list.append(test_error/num_batches)
        dict = {"test_error_list": np.array(test_error_list), "test_logits_epochs": np.stack(test_logits_epochs)}
        if result_filename != 'None':
            sio.savemat(result_filename, dict)
    mean_test_error = np.array(test_error_list).mean()
    std_test_error = np.array(test_error_list).std()
    print("Mean test error: %.2f%%" % mean_test_error)
    print("Std test error: %.2f%%" % std_test_error)
    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")
    
