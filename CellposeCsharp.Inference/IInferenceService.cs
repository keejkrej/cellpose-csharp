using System;
using System.Threading.Tasks;
using OpenCvSharp;

namespace CellposeCsharp.Inference
{
    public interface IInferenceService
    {
        Task<byte[]> RunInferenceAsync(byte[] imageData);
    }
}

