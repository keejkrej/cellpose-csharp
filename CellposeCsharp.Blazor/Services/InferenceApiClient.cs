using System.Net.Http.Headers;
using CellposeCsharp.Blazor.Models;

namespace CellposeCsharp.Blazor.Services;

public class InferenceApiClient
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<InferenceApiClient> _logger;

    public string BaseUrl => _httpClient.BaseAddress?.ToString().TrimEnd('/') ?? "not configured";

    public InferenceApiClient(HttpClient httpClient, ILogger<InferenceApiClient> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<InferenceResult> RunInferenceAsync(
        string fileName,
        Stream imageStream,
        string contentType,
        CancellationToken cancellationToken = default)
    {
        var normalizedContentType = string.IsNullOrWhiteSpace(contentType)
            ? "application/octet-stream"
            : contentType;

        using var content = new MultipartFormDataContent();
        var fileContent = new StreamContent(imageStream);
        fileContent.Headers.ContentType = MediaTypeHeaderValue.Parse(normalizedContentType);
        content.Add(fileContent, "file", fileName);

        var response = await _httpClient.PostAsync("api/Inference", content, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken);
            _logger.LogWarning("Inference call failed with {StatusCode}: {Body}", response.StatusCode, body);
            throw new HttpRequestException($"API returned {(int)response.StatusCode} {response.ReasonPhrase}: {body}");
        }

        var resultBytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        var resultContentType = response.Content.Headers.ContentType?.MediaType ?? "image/png";
        return new InferenceResult(resultBytes, resultContentType);
    }
}
