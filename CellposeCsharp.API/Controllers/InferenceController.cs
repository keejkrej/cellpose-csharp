using Microsoft.AspNetCore.Mvc;
using CellposeCsharp.Inference;

namespace CellposeCsharp.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class InferenceController : ControllerBase
    {
        private readonly IInferenceService _inferenceService;

        public InferenceController(IInferenceService inferenceService)
        {
            _inferenceService = inferenceService;
        }

        [HttpPost]
        public async Task<IActionResult> Infer(IFormFile file)
        {
            if (file == null || file.Length == 0)
                return BadRequest("No file uploaded.");

            using var memoryStream = new MemoryStream();
            await file.CopyToAsync(memoryStream);
            var imageData = memoryStream.ToArray();

            try
            {
                var result = await _inferenceService.RunInferenceAsync(imageData);
                return File(result, "image/png");
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
        }
    }
}

