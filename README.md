# Cellpose-SAM Inference Server (C#)

This project provides an ASP.NET Core Web API backend for running Cellpose-SAM inference using NVIDIA GPUs. It utilizes **ONNX Runtime** with the CUDA execution provider for hardware acceleration.

## Prerequisites

1. **NVIDIA GPU**: Compute capability 6.0 or higher recommended.
2. **Drivers**: Latest NVIDIA drivers installed on the host.
3. **CUDA & cuDNN**:
    * The project is configured for ONNX Runtime GPU 1.23.2.
    * Ensure compatible CUDA (v12.x) and cuDNN (v9.x) libraries are installed.
4. **Models**:
    * Place your Cellpose/SAM ONNX models in the `models/` directory.
    * Update the `ModelPath` in `appsettings.json` or via environment variables if the filename differs.

## Project Structure

* **CellposeCsharp.API**: ASP.NET Core Web API entry point.
* **CellposeCsharp.Inference**: Library containing the inference logic, ONNX Runtime wrappers, and OpenCV preprocessing.

## Running locally (Windows/Linux)

1. Ensure the CUDA/cuDNN DLLs/libraries are in your system PATH or the application output directory.
2. Run the API:

    ```bash
    dotnet run --project CellposeCsharp.API
    ```

3. The API will be available at `http://localhost:5000` (or similar, check logs).
4. Swagger UI: `http://localhost:5000/swagger`

## Running with Docker (Linux Container)

The `Dockerfile` builds a self-contained application on top of the `nvidia/cuda:12.2.2-runtime-ubuntu22.04` image.

1. **Install NVIDIA Container Toolkit** on your host machine to allow Docker containers to access the GPU.
2. Build the image:

    ```bash
    docker build -t cellpose-csharp .
    ```

3. Run the container with GPU access:

    ```bash
    docker run --gpus all -p 8080:8080 -v $(pwd)/models:/app/models cellpose-csharp
    ```

## API Usage

### POST /api/Inference

Upload an image file to run inference.

* **Body**: `multipart/form-data` with a file field (e.g., `file`).
* **Response**: The processed image (currently returns the input image as a placeholder test).

## TODOs for Implementation

1. **Model Export**: Convert your specific Cellpose-SAM Python models to ONNX format.
2. **Preprocessing**: Implement the specific image normalization and resizing logic in `CellposeInferenceService.cs` matching the Python training pipeline.
3. **Postprocessing**: Convert the model output tensors (flows/masks) back into label images.
