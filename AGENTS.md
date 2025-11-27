Purpose
- Notes for maintainers/agents on build/runtime quirks, especially around OpenCvSharp.

Project quick facts
- .NET 9; API at `CellposeCsharp.API`, inference lib at `CellposeCsharp.Inference`.
- OpenCvSharp 4.11.0.20250507; uses official runtime package for Linux/Windows.
- Docker target: `nvidia/cuda:12.2.2-runtime-ubuntu22.04`. Extra native deps installed there: `libgomp1 libicu70 libssl3 libgl1-mesa-glx libglib2.0-0 libgtk2.0-0 libtesseract4 libtiff5 libopenexr25`.
- Default model path: `models/cellpose_sam.onnx` (relative to repo root from API).

Running
- Local: `dotnet run --project CellposeCsharp.API` (ensure GPU/CUDA/cuDNN available if using GPU provider). Launch profile uses `https://localhost:5001;http://localhost:5000`.
- MIGraphX native binaries are **not** committed; drop your MIGraphX-enabled `libonnxruntime.so*` and provider `.so` under `native/onnxruntime-migraphx/` (gitignored) and set  
  `LD_LIBRARY_PATH="$(pwd)/native/onnxruntime-migraphx:/opt/rocm/lib/migraphx/lib:/opt/rocm/lib:$LD_LIBRARY_PATH"`  
  before running so the MIGraphX-enabled `libonnxruntime.so` is loaded instead of the NuGet CPU copy.
- Docker: `docker build -t cellpose-csharp .` then `docker run --gpus all -p 8080:8080 -v $(pwd)/models:/app/models cellpose-csharp`.

GPU providers
- Windows: set `USE_CUDA=1` to enable CUDA provider.
- Linux: set `USE_MIGRAPHX=1` to attempt the MIGraphX provider. The official `Microsoft.ML.OnnxRuntime` NuGet **does not ship** a MIGraphX-enabled `libonnxruntime.so`, so you must override it with a custom build. If MIGraphX isn’t present, it falls back to CPU.
  - Build ORT with MIGraphX (example, ORT repo root):  
    `./build.sh --config Release --build_shared_lib --parallel --skip_tests --use_migraphx --compile_no_warning_as_error --cmake_extra_defines CMAKE_PREFIX_PATH=/opt/rocm --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF --build_dir build/Linux`  
    (Requires ROCm with MIGraphX at `/opt/rocm`; `libmigraphx*.so` live under `/opt/rocm/lib/migraphx/lib`.)
  - Drop the MIGraphX-enabled binaries under `native/onnxruntime-migraphx/` locally (gitignored) and set `LD_LIBRARY_PATH` to include that path (plus ROCm) so they’re loaded at runtime.
  - Set `LD_LIBRARY_PATH="/opt/rocm/lib/migraphx/lib:/opt/rocm/lib:$LD_LIBRARY_PATH"` when running so MIGraphX deps resolve. The managed NuGet (currently 1.23.2) can load a newer native (e.g. ORT 1.24.x) but keep an eye on version drift.

OpenCvSharp on Arch Linux
- The official runtime package is built for Ubuntu 22.04 and expects SONAMEs like `libtesseract.so.4`, `libtiff.so.5`, etc. On Arch this causes `NativeMethods` init failures unless you replace the native library.
- Solution: rebuild `libOpenCvSharpExtern.so` against Arch system libraries and override the NuGet-provided one.
  1) Install deps (rough set): `sudo pacman -S base-devel opencv gtk2 tesseract openexr libtiff openjpeg2 gdk-pixbuf2 vtk hdf5 gstreamer` (matches Arch’s OpenCV optional modules so `ldd` is clean).
  2) Build extern:
     ```
     git clone --depth 1 https://github.com/shimat/opencvsharp.git
     cd opencvsharp && git fetch --tags --depth 1 && git checkout tags/4.11.0.20250507
     mkdir build && cd build
     cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_POLICY_VERSION_MINIMUM=3.5 ../src
     cmake --build . --config Release -j
     ```
     Result: `build/OpenCvSharpExtern/libOpenCvSharpExtern.so`.
  3) Override NuGet runtime copy (example path):
     ```
     cp build/OpenCvSharpExtern/libOpenCvSharpExtern.so ~/.nuget/packages/opencvsharp4.official.runtime.linux-x64/4.11.0.20250507/runtimes/linux-x64/native/
     ```
     Keep a backup of the original if needed.
  4) Verify: `ldd ~/.nuget/.../libOpenCvSharpExtern.so` should show no “not found” entries.
- NuGet restore may re-drop the original; re-copy the rebuilt `.so` after restores or set `LD_LIBRARY_PATH` to the rebuilt location when running locally.

Container note
- Inside the Docker image (Ubuntu 22.04) the official runtime works as-is with the apt packages above; no rebuild required there.
