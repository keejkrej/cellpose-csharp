# Stage 1: Build
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy solution and project files
COPY ["CellposeCsharp.sln", "./"]
COPY ["CellposeCsharp.API/CellposeCsharp.API.csproj", "CellposeCsharp.API/"]
COPY ["CellposeCsharp.Inference/CellposeCsharp.Inference.csproj", "CellposeCsharp.Inference/"]

# Restore dependencies
RUN dotnet restore

# Copy the rest of the code
COPY . .

# Build and publish (Self-contained to run on the NVIDIA base image easily)
RUN dotnet publish "CellposeCsharp.API/CellposeCsharp.API.csproj" -c Release -o /app/publish --self-contained -r linux-x64

# Stage 2: Runtime
# We use the NVIDIA CUDA runtime image to ensure GPU libraries are available.
# Ensure the CUDA version matches what ONNX Runtime expects (approx CUDA 12.x for recent ORT).
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS final
WORKDIR /app

# Install system dependencies required by .NET and OpenCV
# libgomp1 is often needed by ONNX Runtime/OpenCV
# libicu and libssl are needed by .NET
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libicu70 \
    libssl3 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/publish .

# Expose the port
EXPOSE 8080

# Set environment variables
ENV ASPNETCORE_URLS=http://+:8080
# Ensure ONNX Runtime finds CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["./CellposeCsharp.API"]

