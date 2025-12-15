# import torch

# print("PyTorch 버전:", torch.__version__)
# print("CUDA 사용 가능?", torch.cuda.is_available())
# print("GPU 개수:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("0번 GPU 이름:", torch.cuda.get_device_name(0))

import tensorflow as tf
print("====")
print("TensorFlow:", tf.__version__)
print("GPU 리스트:", tf.config.list_physical_devices('GPU'))

print("CUDA로 빌드됨?", tf.test.is_built_with_cuda())
print("GPU 사용 가능?", tf.test.is_gpu_available())
