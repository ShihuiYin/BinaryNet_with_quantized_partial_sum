# BinaryNet_with_quantized_partial_sum
test_BNN_cifar10_monte_carlo_C3SRAM_Arch.py is the main file for modeling the effect of ADC offsets and bitline noise on the BNN accuracy for a given SRAM depth (256), quantization scheme (scaled linear).

To see the baseline accuracy of model 128C3-128C3-MP2-256C3-256C3-MP2-512C3-512C3-MP2-1024FC-1024FC-10:
python test_BNN_cifar10_monte_carlo_C3SRAM_Arch.py -ac "128-256-512-1024" -bl
We should get test error of 11.53%.

To see the baseline accuracy of model 128C3-128C3-MP2-256C3-256C3-MP2-256C3-256C3-MP2-512FC-512FC-10:
python test_BNN_cifar10_monte_carlo_C3SRAM_Arch.py -ac "128-256-256-512" -bl
We should get test error of 11.93%.
