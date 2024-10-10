# Use Onnxruntime with python

## 1. Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11) & [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10)

## 2. Install [python package manager](https://www.anaconda.com/download/success)
2.1 create environment 'AIEnv'.<br>
2.2 excute pip in the environment cmd (conda activate AIEnv).<br>
2.3 vscode choose the environment to excute.<br>

## 3. Install onnxruntime
> pip install onnxruntime<br>
> (excute in AIEnv environment)<br>
```python
import onnxruntime as ort
session = ort.InferenceSession(model_file_path)
# session.run(...)
```

## 5. Run with GPU
### 1. DirectML
> pip install onnxruntime-directml
```python
import onnxruntime
session = onnxruntime.InferenceSession(model_file_path, providers=['DmlExecutionProvider'])
print("onnxruntime providers:",onnxruntime.get_device(),onnxruntime.get_available_providers())
print("session providers:", session.get_providers())
```
> expect:<br>
> onnxruntime providers: CPU-DML ['DmlExecutionProvider', 'CPUExecutionProvider']<br>
> session providers: ['DmlExecutionProvider', 'CPUExecutionProvider']<br>

> case: onnxruntime providers *DO NOT* contains DmlExecutionProvider.<br>
> fix:(reinstall packages)<br>
   > pip uninstall onnxruntime-gpu onnxruntime-directml onnxruntime<br>
   > pip install onnxruntime<br>
   > pip install onnxruntime-directml<br>

### 2. CUDA
> pip install onnxruntime-gpu
```python
import onnxruntime
session = onnxruntime.InferenceSession(model_file_path, providers=['CUDAExecutionProvider'])
print("onnxruntime providers:",onnxruntime.get_device(),onnxruntime.get_available_providers())
print("session providers:", session.get_providers())
```
> expect:<br>
> onnxruntime providers: GPU ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']<br>
> session providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']<br>

> case1: onnxruntime providers *DO NOT* contains CUDAExecutionProvider.<br>
> fix: reinstall packages as before<br>

> case2: only session providers *DO NOT* contains CUDAExecutionProvider.<br>
> fix: install and import torch <br>

## 6. Install [pytorch](https://pytorch.org/)
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124<br>
> (excute in AIEnv environment)<br>
```python
import torch
print(torch.cuda.is_available(), torch.backends.cudnn.version())
```
> expect: True 90100

## 6.Sameple python code:

```python
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

> expect:<br>
> Available providers: ['CPUExecutionProvider']<br>
> Current provider: {'CPUExecutionProvider': {}}<br>
> Top 10 predictions for ResNet50 v2...<br>
> Label: golden retriever, Confidence: 0.8391<br>
> Label: kuvasz, Confidence: 0.0905<br>
> Label: otterhound, Confidence: 0.0135<br>
> Label: clumber, Confidence: 0.0101<br>
> Label: Sussex spaniel, Confidence: 0.0072<br>
> Label: Labrador retriever, Confidence: 0.0060<br>
> Label: Tibetan terrier, Confidence: 0.0035<br>
> Label: Great Pyrenees, Confidence: 0.0028<br>
> Label: Saluki, Confidence: 0.0027<br>
> Label: English setter, Confidence: 0.0021<br>
