# Use Onnxruntime with C#

## 1. Install [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11) & [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=10)

## 2. Download demo [onnxruntime.csharp.resnet50](https://github.com/microsoft/onnxruntime/tree/main/csharp/sample/Microsoft.ML.OnnxRuntime.ResNet50v2Sample)
2.1 Download onnx model [resnet50-v2-7.onnx](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx)<br>
2.2 Set modelFilePath & imageFilePath.<br>
2.3 Run and expect result:<br>
> Top 10 predictions for ResNet50 v2...<br>
> Label: Golden Retriever, Confidence: 0.7005542<br>
> Label: Kuvasz, Confidence: 0.17308559<br>
> Label: Otterhound, Confidence: 0.019904016<br>
> Label: Clumber Spaniel, Confidence: 0.018665986<br>
> Label: Saluki, Confidence: 0.0110780625<br>
> Label: Sussex Spaniel, Confidence: 0.0073013026<br>
> Label: Labrador Retriever, Confidence: 0.0070578177<br>
> Label: Pyrenean Mountain Dog, Confidence: 0.00638076<br>
> Label: Tibetan Terrier, Confidence: 0.006171986<br>
> Label: English Setter, Confidence: 0.004174494<br>

## 3. Run with GPU

Add InferenceSessionOptions

### 1. DirectML

Add Code:

```csharp
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_DML();
var session = new InferenceSession(modelFilePath, sessionOptions);
```

> Exception:<br>
> 'Unable to find an entry point named 'OrtSessionOptionsAppendExecutionProvider_DML' in DLL 'onnxruntime'.<br>

> Fix:<br>
> `remove Microsoft.ML.OnnxRuntime package`<br>
> `nuget install Microsoft.ML.OnnxRuntime.DirectML`<br>

### 2. CUDA

Add Code:

```csharp
var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider_CUDA();
var session = new InferenceSession(modelFilePath, sessionOptions);
```

> Exception1:<br>
> Unable to find an entry point named 'OrtSessionOptionsAppendExecutionProvider_CUDA' in DLL 'onnxruntime'.<br>

> Fix:<br>
> `remove Microsoft.ML.OnnxRuntime`<br>
>    `nuget install Microsoft.ML.OnnxRuntime.Gpu.Windows` or `nuget install Microsoft.ML.OnnxRuntime.Gpu`<br>

--

> Exception2:<br>
> Microsoft.ML.OnnxRuntime.OnnxRuntimeException:<br>
> '[ErrorCode:RuntimeException] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "C:\Dev\AI\OnnxRuntimeResNet50\bin\Debug\net8.0\runtimes\win-x64\native\onnxruntime_providers_cuda.dll"<br>

> Fix:<br>
> `nuget install TorchSharp-cuda-windows`<br>
> add code:<br>
> `Console.WriteLine(TorchSharp.torch.cuda.is_cudnn_available());`
> `Console.WriteLine(TorchSharp.torch.cuda.is_available());`
