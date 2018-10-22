using System;
using System.IO;
using System.Threading.Tasks;
using e2eClassification.Models;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;

namespace e2eClassification
{
    class Program
    {
        static readonly string _dataPath = Path.Combine("C:\\src\\leeds", "data", "iris.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "trained", "IrisClusteringModel.zip");
        static async Task Main(string[] args)
        {
            PredictionModel<IrisData, ClusterPrediction> model = Train();
            await model.WriteAsync(_modelPath);
            
            var prediction = model.Predict(TestIrisData.Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
        private static PredictionModel<IrisData, ClusterPrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<IrisData>(separator: ','));
            pipeline.Add(new ColumnConcatenator(
                    "Features",
                    "SepalLength",
                    "SepalWidth",
                    "PetalLength",
                    "PetalWidth"));
            pipeline.Add(new KMeansPlusPlusClusterer() { K = 3 });
            
            var model = pipeline.Train<IrisData, ClusterPrediction>();
            return model;
        }
    }
}
