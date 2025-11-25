using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;

namespace CellposeCsharp.Inference
{
    public class CellposeInferenceService : IInferenceService, IDisposable
    {
        private InferenceSession? _session;
        private readonly string _modelPath;

        public CellposeInferenceService(string modelPath)
        {
            _modelPath = modelPath;
            InitializeSession();
        }

        private void InitializeSession()
        {
            // Check if model exists
            if (!File.Exists(_modelPath))
            {
                // For now, we won't throw if model is missing to allow app to start in dev
                // In production, this should probably throw or log an error
                Console.WriteLine($"Warning: Model file not found at {_modelPath}");
                return;
            }

            var sessionOptions = new SessionOptions();

            // Configure CUDA provider
            // Note: The CUDA provider DLLs need to be in the output directory.
            // Ensure you have installed CUDA Toolkit and cuDNN compatible with the ONNX Runtime version.
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load CUDA provider: {ex.Message}. Falling back to CPU.");
            }

            _session = new InferenceSession(_modelPath, sessionOptions);
        }

        public async Task<byte[]> RunInferenceAsync(byte[] imageData)
        {
            if (_session == null)
            {
                throw new InvalidOperationException("Inference session is not initialized. Model may be missing.");
            }

            // 1. Preprocess image (decoding, resizing, normalizing) using OpenCvSharp
            using var originalMat = Mat.FromImageData(imageData, ImreadModes.Color);
            
            // Resize to model input size (256x256)
            const int modelInputSize = 256;
            using var resizedMat = new Mat();
            Cv2.Resize(originalMat, resizedMat, new OpenCvSharp.Size(modelInputSize, modelInputSize));
            
            // Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
            using var rgbMat = new Mat();
            Cv2.CvtColor(resizedMat, rgbMat, ColorConversionCodes.BGR2RGB);
            
            // Normalize: Convert to float32 and normalize to [0, 1] range
            // Cellpose models typically expect normalized input
            rgbMat.ConvertTo(rgbMat, MatType.CV_32F, 1.0 / 255.0);
            
            // Extract image data and rearrange from HWC to CHW format
            var imageDataArray = new float[3 * modelInputSize * modelInputSize];
            unsafe
            {
                var dataPtr = (float*)rgbMat.DataPointer;
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < modelInputSize; h++)
                    {
                        for (int w = 0; w < modelInputSize; w++)
                        {
                            int srcIdx = h * modelInputSize * 3 + w * 3 + c;
                            int dstIdx = c * modelInputSize * modelInputSize + h * modelInputSize + w;
                            imageDataArray[dstIdx] = dataPtr[srcIdx];
                        }
                    }
                }
            }
            
            // Create input tensor [1, 3, 256, 256]
            var inputTensor = new DenseTensor<float>(imageDataArray, new[] { 1, 3, modelInputSize, modelInputSize });
            
            // Get input name from model metadata
            var inputName = _session.InputMetadata.Keys.First();
            var inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            // 2. Run Inference
            using var results = _session.Run(inputContainer);
            
            // Get output tensors
            // Model returns: output (flows+cellprob) and style
            var outputTensor = results.First().AsTensor<float>();
            var outputShape = outputTensor.Dimensions.ToArray();
            
            if (outputShape.Length != 4 || outputShape[1] < 2)
            {
                throw new InvalidOperationException($"Unexpected model output shape: [{string.Join(",", outputShape)}]");
            }

            var outputHeight = outputShape[2];
            var outputWidth = outputShape[3];
            var channelSize = outputHeight * outputWidth;
            
            // 3. Postprocess: Convert flow vectors to a colored flow field map
            // Channels: 0=Y flow, 1=X flow, 2=Cell probability (unused here)
            var outputArray = outputTensor.ToArray();
            var flowY = new float[channelSize];
            var flowX = new float[channelSize];
            for (int i = 0; i < channelSize; i++)
            {
                flowY[i] = outputArray[i]; // channel 0
                flowX[i] = outputArray[channelSize + i]; // channel 1
            }

            using var flowYMat = new Mat(outputHeight, outputWidth, MatType.CV_32F);
            using var flowXMat = new Mat(outputHeight, outputWidth, MatType.CV_32F);
            unsafe
            {
                var flowYPtr = (float*)flowYMat.DataPointer;
                var flowXPtr = (float*)flowXMat.DataPointer;
                for (int i = 0; i < channelSize; i++)
                {
                    flowYPtr[i] = flowY[i];
                    flowXPtr[i] = flowX[i];
                }
            }

            using var magnitude = new Mat();
            using var angle = new Mat();
            Cv2.CartToPolar(flowXMat, flowYMat, magnitude, angle, angleInDegrees: true);

            // Normalize magnitude to [0,1] for value channel; keep hue from angle (0-180 in OpenCV HSV)
            using var magnitudeNormalized = new Mat();
            Cv2.Normalize(magnitude, magnitudeNormalized, 0.0, 1.0, NormTypes.MinMax);

            using var angleHue = new Mat();
            angle.ConvertTo(angleHue, MatType.CV_32F, 0.5); // scale 0-360 -> 0-180 for OpenCV HSV

            using var saturation = new Mat(outputHeight, outputWidth, MatType.CV_32F, Scalar.All(1.0));
            using var hsv = new Mat();
            Cv2.Merge(new Mat[] { angleHue, saturation, magnitudeNormalized }, hsv);

            using var flowBgr = new Mat();
            Cv2.CvtColor(hsv, flowBgr, ColorConversionCodes.HSV2BGR);

            using var flowBgr8U = new Mat();
            flowBgr.ConvertTo(flowBgr8U, MatType.CV_8UC3, 255.0);

            // Resize flow map back to the original image dimensions
            using var flowResized = new Mat();
            Cv2.Resize(flowBgr8U, flowResized, new OpenCvSharp.Size(originalMat.Width, originalMat.Height));

            // Encode as PNG (flow field visualization)
            Cv2.ImEncode(".png", flowResized, out byte[] resultBytes);
            
            return await Task.FromResult(resultBytes);
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

