# Use Onnxruntime with python

## 1. Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11) & [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10)

## 2. Install [python package manager](https://www.anaconda.com/download/success)
2.1 create environment 'AIEnv'.
2.2 excute pip in the environment cmd (conda activate AIEnv).
2.3 vscode choose the environment to excute.

## 3. Install onnxruntime
> pip install onnxruntime
> (excute in AIEnv environment)
```
import onnxruntime as ort
session = ort.InferenceSession(model_file_path)
# session.run(...)
```

## 5. Run with GPU
### 1. DirectML
> pip install onnxruntime-directml
```
import onnxruntime
session = onnxruntime.InferenceSession(model_file_path, providers=['DmlExecutionProvider'])
print("onnxruntime providers:",onnxruntime.get_device(),onnxruntime.get_available_providers())
print("session providers:", session.get_providers())
```
> expect:
> onnxruntime providers: CPU-DML ['DmlExecutionProvider', 'CPUExecutionProvider']
> session providers: ['DmlExecutionProvider', 'CPUExecutionProvider']

> case: onnxruntime providers *DO NOT* contains DmlExecutionProvider.
> fix:(reinstall packages)
   > pip uninstall onnxruntime-gpu
   > pip uninstall onnxruntime-directml
   > pip uninstall onnxruntime
   > pip install onnxruntime
   > pip install onnxruntime-directml

### 2. CUDA
> pip install onnxruntime-gpu
```
import onnxruntime
session = onnxruntime.InferenceSession(model_file_path, providers=['CUDAExecutionProvider'])
print("onnxruntime providers:",onnxruntime.get_device(),onnxruntime.get_available_providers())
print("session providers:", session.get_providers())
```
> expect:
> onnxruntime providers: GPU ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
> session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']

> case1: onnxruntime providers *DO NOT* contains CUDAExecutionProvider.
> fix: reinstall packages as before

> case2: only session providers *DO NOT* contains CUDAExecutionProvider.
> fix: install and import torch 

## 6. Install [pytorch](https://pytorch.org/)
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> (excute in AIEnv environment)
```
import torch
print(torch.cuda.is_available(), torch.backends.cudnn.version())
```
> expect: True 90100

## 6.Sameple python code:

```
import onnxruntime as ort
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet50_Weights

model_file_path = r"resnet50-v2-7.onnx"
image_file_path = r"dog.jpeg"

# Create ONNX runtime session
session = ort.InferenceSession(model_file_path)
print("Available providers:", session.get_providers())
print("Current provider:", session.get_provider_options())

# Read and preprocess image
image = Image.open(image_file_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Run inference
ort_inputs = {session.get_inputs()[0].name: input_batch.numpy()}
ort_outputs = session.run(None, ort_inputs)

# Postprocess to get softmax vector
output = ort_outputs[0]
softmax = torch.nn.functional.softmax(torch.tensor(output), dim=1)

# Extract top 10 predicted classes
top10 = torch.topk(softmax, 10)

# Get label mapping
weights = ResNet50_Weights.DEFAULT
labels = weights.meta["categories"]

# Print results to console
print("Top 10 predictions for ResNet50 v2...")
print("--------------------------------------------------------------")
for i in range(10):
    print(f"Label: {labels[top10.indices[0][i]]}, Confidence: {top10.values[0][i].item():.4f}")
```

> expect:
> Available providers: ['CPUExecutionProvider']
> Current provider: {'CPUExecutionProvider': {}}
> Top 10 predictions for ResNet50 v2...
> Label: golden retriever, Confidence: 0.8391
> Label: kuvasz, Confidence: 0.0905
> Label: otterhound, Confidence: 0.0135
> Label: clumber, Confidence: 0.0101
> Label: Sussex spaniel, Confidence: 0.0072
> Label: Labrador retriever, Confidence: 0.0060
> Label: Tibetan terrier, Confidence: 0.0035
> Label: Great Pyrenees, Confidence: 0.0028
> Label: Saluki, Confidence: 0.0027
> Label: English setter, Confidence: 0.0021
