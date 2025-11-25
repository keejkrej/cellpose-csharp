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
            
            // 3. Postprocess: Convert output to image
            // Output shape should be [1, 3, 256, 256] for flows+cellprob
            // Channels: 0=Y flow, 1=X flow, 2=Cell probability
            var outputArray = outputTensor.ToArray();
            var cellProbChannel = 2; // Cell probability is the last channel
            
            // Extract cell probability channel
            int channelSize = modelInputSize * modelInputSize;
            var probValues = new float[channelSize];
            for (int i = 0; i < channelSize; i++)
            {
                probValues[i] = outputArray[cellProbChannel * channelSize + i];
            }
            
            // Apply sigmoid activation if values are logits (typical for Cellpose)
            // Find min/max to determine if we need sigmoid
            float minProb = probValues.Min();
            float maxProb = probValues.Max();
            
            // If values are outside [0,1], they're likely logits - apply sigmoid
            if (minProb < 0 || maxProb > 1)
            {
                for (int i = 0; i < channelSize; i++)
                {
                    // Sigmoid: 1 / (1 + exp(-x))
                    probValues[i] = 1.0f / (1.0f + (float)Math.Exp(-probValues[i]));
                }
            }
            
            // Create probability map
            using var probMat = new Mat(modelInputSize, modelInputSize, MatType.CV_32F);
            unsafe
            {
                var probPtr = (float*)probMat.DataPointer;
                for (int i = 0; i < channelSize; i++)
                {
                    probPtr[i] = probValues[i];
                }
            }
            
            // Normalize to [0, 255] for visualization
            using var normalizedMat = new Mat();
            probMat.ConvertTo(normalizedMat, MatType.CV_8U, 255.0);
            
            // Resize probability map to original image size
            using var probResized = new Mat();
            Cv2.Resize(normalizedMat, probResized, new OpenCvSharp.Size(originalMat.Width, originalMat.Height));
            
            // Create colored visualization for the probability heatmap (no overlay)
            using var coloredProb = new Mat();
            Cv2.ApplyColorMap(probResized, coloredProb, ColormapTypes.Jet);

            // Encode as PNG (heatmap only)
            Cv2.ImEncode(".png", coloredProb, out byte[] resultBytes);
            
            return await Task.FromResult(resultBytes);
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}

