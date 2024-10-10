# Use Onnxruntime with C#

## 1. Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11) & [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10)

## 2. Download demo [onnxruntime.csharp.resnet50(github.com)](https://github.com/microsoft/onnxruntime/tree/main/csharp/sample/Microsoft.ML.OnnxRuntime.ResNet50v2Sample)
2.1 Download onnx model [resnet50-v2-7.onnx (github.com)](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx)
2.2 Set modelFilePath & imageFilePath.
2.3 Run and expect result:
> Top 10 predictions for ResNet50 v2...
> Label: Golden Retriever, Confidence: 0.7005542
> Label: Kuvasz, Confidence: 0.17308559
> Label: Otterhound, Confidence: 0.019904016
> Label: Clumber Spaniel, Confidence: 0.018665986
> Label: Saluki, Confidence: 0.0110780625
> Label: Sussex Spaniel, Confidence: 0.0073013026
> Label: Labrador Retriever, Confidence: 0.0070578177
> Label: Pyrenean Mountain Dog, Confidence: 0.00638076
> Label: Tibetan Terrier, Confidence: 0.006171986
> Label: English Setter, Confidence: 0.004174494

## 3. Run with GPU

Add InferenceSessionOptions

### 1. DirectML

Add Code:

```
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_DML();
var session = new InferenceSession(modelFilePath, sessionOptions);
```

> Exception:
> 'Unable to find an entry point named 'OrtSessionOptionsAppendExecutionProvider_DML' in DLL 'onnxruntime'.

> Fix:
> remove Microsoft.ML.OnnxRuntime package
> nuget install Microsoft.ML.OnnxRuntime.DirectML

### 2. CUDA

Add Code:

```
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_CUDA();
var session = new InferenceSession(modelFilePath, sessionOptions);
```

> Exception1:
> Unable to find an entry point named 'OrtSessionOptionsAppendExecutionProvider_CUDA' in DLL 'onnxruntime'.

> Fix:
> remove Microsoft.ML.OnnxRuntime
> nuget install Microsoft.ML.OnnxRuntime.Gpu.Windows
> or nuget install Microsoft.ML.OnnxRuntime.Gpu

--

> Exception2:
> Microsoft.ML.OnnxRuntime.OnnxRuntimeException:
> '[ErrorCode:RuntimeException] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "C:\Dev\AI\OnnxRuntimeResNet50\bin\Debug\net8.0\runtimes\win-x64\native\onnxruntime_providers_cuda.dll"

> Fix:
> nuget install TorchSharp-cuda-windows
> add code:
> `Console.WriteLine(TorchSharp.torch.cuda.is_cudnn_available());`
> `Console.WriteLine(TorchSharp.torch.cuda.is_available());`
