using CellposeCsharp.Inference;
using System.IO;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Register Inference Service
// Resolve model path relative to content root
var modelPathConfig = builder.Configuration.GetValue<string>("ModelPath") ?? "models/cellpose_sam.onnx";
var contentRoot = builder.Environment.ContentRootPath;
var modelPath = Path.IsPathRooted(modelPathConfig) 
    ? modelPathConfig 
    : Path.Combine(contentRoot, "..", modelPathConfig.Replace('/', Path.DirectorySeparatorChar));
modelPath = Path.GetFullPath(modelPath); // Resolve any .. or . in path
Console.WriteLine($"Loading model from: {modelPath}");
builder.Services.AddSingleton<IInferenceService>(sp => new CellposeInferenceService(modelPath));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
