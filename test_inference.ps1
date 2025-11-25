# Test script for Cellpose Inference API
$apiUrl = "http://localhost:5027/api/Inference"

# Create a simple test image using .NET (or use an existing image)
# For now, let's check if we can find any test images, or create one

Write-Host "Testing Cellpose Inference API at $apiUrl"
Write-Host ""

# Check if API is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5027/swagger/index.html" -Method GET -UseBasicParsing -ErrorAction Stop
    Write-Host "✓ API is running (Status: $($response.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "✗ API is not accessible: $_" -ForegroundColor Red
    exit 1
}

# Create a simple test image using System.Drawing
Add-Type -AssemblyName System.Drawing

$width = 512
$height = 512
$bitmap = New-Object System.Drawing.Bitmap($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)

# Fill with a gradient pattern
for ($y = 0; $y -lt $height; $y++) {
    for ($x = 0; $x -lt $width; $x++) {
        $colorValue = [int](($x + $y) / 2) % 256
        $color = [System.Drawing.Color]::FromArgb($colorValue, $colorValue, $colorValue)
        $bitmap.SetPixel($x, $y, $color)
    }
}

# Add some circles to simulate cells
$brush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::White)
$graphics.FillEllipse($brush, 100, 100, 80, 80)
$graphics.FillEllipse($brush, 300, 200, 100, 100)
$graphics.FillEllipse($brush, 150, 350, 90, 90)

# Save to memory stream
$memoryStream = New-Object System.IO.MemoryStream
$bitmap.Save($memoryStream, [System.Drawing.Imaging.ImageFormat]::Png)
$imageBytes = $memoryStream.ToArray()
$memoryStream.Close()
$graphics.Dispose()
$bitmap.Dispose()

Write-Host "Created test image (${width}x${height}px)" -ForegroundColor Cyan

# Send to API
Write-Host "Sending image to inference endpoint..." -ForegroundColor Cyan
try {
    $boundary = [System.Guid]::NewGuid().ToString()
    $bodyLines = @(
        "--$boundary",
        "Content-Disposition: form-data; name=`"file`"; filename=`"test.png`"",
        "Content-Type: image/png",
        "",
        [System.Text.Encoding]::GetEncoding("iso-8859-1").GetString($imageBytes),
        "--$boundary--"
    )
    
    $body = $bodyLines -join "`r`n"
    $bodyBytes = [System.Text.Encoding]::GetEncoding("iso-8859-1").GetBytes($body)
    
    $response = Invoke-WebRequest -Uri $apiUrl -Method POST -Body $bodyBytes -ContentType "multipart/form-data; boundary=$boundary" -ErrorAction Stop
    
    Write-Host "✓ Inference successful! (Status: $($response.StatusCode))" -ForegroundColor Green
    Write-Host "Response size: $($response.Content.Length) bytes" -ForegroundColor Cyan
    
    # Save the result
    $outputPath = "test_output.png"
    [System.IO.File]::WriteAllBytes($outputPath, $response.Content)
    Write-Host "✓ Saved result to: $outputPath" -ForegroundColor Green
    
} catch {
    Write-Host "✗ Inference failed: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Error details: $responseBody" -ForegroundColor Yellow
    }
    exit 1
}
