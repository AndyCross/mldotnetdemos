using Microsoft.ML.Runtime.Api;

namespace e2eTensorflow.ImageData
{
    public class ImageNetPrediction
    {
        [ColumnName("Score")]
        public float[] PredictedLabels;
    }
}