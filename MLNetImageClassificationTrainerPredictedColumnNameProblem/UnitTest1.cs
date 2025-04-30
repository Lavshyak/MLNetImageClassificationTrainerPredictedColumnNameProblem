using Microsoft.ML;
using Microsoft.ML.Vision;
using SkiaSharp;

namespace MLNetImageClassificationTrainerPredictedColumnNameProblem;

public class DataInput
{
    public byte[] SourceImageBytes { get; set; } = null!;
    public string LabelValue { get; set; } = null!;
}

public class DataOutput : DataInput
{
    public string PredictedLabelValue { get; set; } = null!;
}


public class UnitTest1
{
    [Fact]
    public void HasException()
    {
        var mlContext = new MLContext();
        var (dataInputs, dataInputsDataView) = CreateRandomInputData(mlContext, 2, 2);

        var estimator = GenerateClassificationEstimator(mlContext, "PredictedLabelKey");
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => estimator.Fit(dataInputsDataView));
        Assert.Equal("Could not find input column 'PredictedLabelKey' (Parameter 'inputSchema')", ex.Message);
    }
    
    
    [Fact]
    public void NoException()
    {
        var mlContext = new MLContext();
        var (dataInputs, dataInputsDataView) = CreateRandomInputData(mlContext, 2, 2);

        var estimator = GenerateClassificationEstimator(mlContext, "PredictedLabel");
        estimator.Fit(dataInputsDataView);
    }

    public IEstimator<ITransformer> GenerateClassificationEstimator(MLContext mlContext,
        string predictedLabelKeyColumnName = "PredictedLabel")
    {
        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = nameof(DataInput.SourceImageBytes),
            LabelColumnName = "LabelKey",
            PredictedLabelColumnName = predictedLabelKeyColumnName,
            ValidationSet = null,
            Arch = ImageClassificationTrainer.Architecture.InceptionV3,
            //MetricsCallback = (metrics) => Console.WriteLine(metrics),
            TestOnTrainSet = true,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true
        };

        var trainingPipeline =
            mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(DataInput.LabelValue),
                    outputColumnName: "LabelKey", keyOrdinality: Microsoft.ML.Transforms
                        .ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: predictedLabelKeyColumnName,
                    outputColumnName: nameof(DataOutput.PredictedLabelValue)));

        return trainingPipeline;
    }
    
    public (DataInput[], IDataView) CreateRandomInputData(MLContext mlContext, int imagesCount, int widthAndHeight)
    {
        var random = new Random(1);

        var inputDatas = Enumerable.Range(0, imagesCount).Select(i =>
        {
            var pixelBytes = new byte[widthAndHeight * widthAndHeight * 4];
            random.NextBytes(pixelBytes);
         
            var skImage = SKImage.FromPixels(new SKImageInfo(widthAndHeight, widthAndHeight, SKColorType.Bgra8888), SKData.CreateCopy(pixelBytes));
            var sourceImageBytes = skImage.Encode(SKEncodedImageFormat.Jpeg, 1).ToArray();
            
            var inputData = new DataInput { SourceImageBytes = sourceImageBytes, LabelValue = $"label_{i}" };
            return inputData;
        }).ToArray();

        var inputDatasDataView = mlContext.Data.LoadFromEnumerable(inputDatas);

        return (inputDatas, inputDatasDataView);
    }
}