using System;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.ImageAnalytics;
using Microsoft.ML.Transforms;
using e2eTensorflow.ImageData;
using e2eTensorflow.Model;

namespace e2eTensorflow
{
    class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                if (typeof(TensorFlowTransform) == null) throw new Exception("Tensorflow not loaded correctly");

                if (typeof(ImageLoaderTransform) == null) throw new Exception("ImageAnalytics not loaded correctly");

                var modelBuilder = new ModelTrainer(
                                           ModelHelpers.GetAssetsPath("data", "tags.tsv"),
                                           ModelHelpers.GetAssetsPath("images"),
                                           ModelHelpers.GetAssetsPath("model", "tensorflow_inception_graph.pb"),
                                           ModelHelpers.GetAssetsPath("model", "imageClassifier.zip"));

                await modelBuilder.BuildAndTrain();

                var modelEvaluator = new ModelEvaluator(
                                           ModelHelpers.GetAssetsPath("data", "tags.tsv"),
                                           ModelHelpers.GetAssetsPath("images"),
                                           ModelHelpers.GetAssetsPath("model", "imageClassifier.zip"));

                await modelEvaluator.Evaluate();

            } catch (Exception ex)
            {
                Console.WriteLine("InnerException: {0}", ex.InnerException.ToString());
                throw;
            }

            Console.WriteLine("End of process");
            Console.ReadKey();
        }
    }
}
