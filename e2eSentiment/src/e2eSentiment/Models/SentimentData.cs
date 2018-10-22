// <Snippet1>
using Microsoft.ML.Runtime.Api;
// </Snippet1>

namespace e2eSentiment.Models
{
    public class SentimentData
    {
        public bool Label { get; set; }
        public string Text { get; set; }
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        [ColumnName("Probability")]
        public float Probability { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}