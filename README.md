🔬 GPU Benchmarking
To validate the performance advantage of using a GPU with PyTorch in WSL2, two benchmarking scripts are provided:

benchmark_torch_gpu.py:
A simple test that compares inference time of a ResNet18 model between CPU and GPU using 32 input images.

benchmark_torch_gpu_large.py:
A more robust benchmark using a larger ResNet50 model with 64-image batches and 200 iterations to produce more consistent results.

These scripts help confirm that CUDA acceleration is working correctly and provide a practical speed comparison for deep learning inference.
