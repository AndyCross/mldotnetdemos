using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;
using e2eSentiment.Models;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML;
using Microsoft.ML.Core.Data;

namespace e2eSentiment
{
    class Program
    {
        static readonly string _dataPath = Path.Combine("C:\\src\\leeds", "Data", "wikipedia-detox-250-line-data.tsv");
        static readonly string _testDataPath = Path.Combine("C:\\src\\leeds", "Data", "wikipedia-detox-250-line-test.tsv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "trained", "Model.zip");
        
        static async Task Main(string[] args)
        {
            using (var env = new LocalEnvironment()) {
                IDataView trainingData = GetData(env, _dataPath);

                var pipeline = new TextTransform(env, "Text", "Features")  //Convert the text column to numeric vectors (Features column)   
                                .Append(new LinearClassificationTrainer(env, new LinearClassificationTrainer.Arguments(), 
                                                                        "Features", 
                                                                        "Label"));                                                                
               
                Console.WriteLine("=============== Create and Train the Model ===============");
                var model = pipeline.Fit(trainingData);
                Console.WriteLine("=============== End of training ===============");  
                Console.WriteLine();

                IDataView testData = GetData(env, _testDataPath);

                Console.WriteLine("=============== Evaluating Model's accuracy with Test data===============");
                var predictions = model.Transform(testData);

                var binClassificationCtx = new BinaryClassificationContext(env);
                var metrics = binClassificationCtx.Evaluate(predictions, "Label");

                Console.WriteLine();
                Console.WriteLine("Model quality metrics evaluation");
                Console.WriteLine("------------------------------------------");
                Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
                Console.WriteLine($"Auc: {metrics.Auc:P2}");
                Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
                Console.WriteLine("=============== End of Model's evaluation ===============");
                Console.WriteLine();

                SaveModelAsFile(env, model);
            }

            while(true) {
                string line = Console.ReadLine();
                if (string.IsNullOrEmpty(line)) {
                    break;
                }
                PredictOne(line);
            }
        }

        private static void SaveModelAsFile(LocalEnvironment env, TransformerChain<BinaryPredictionTransformer<Microsoft.ML.Runtime.Internal.Internallearn.IPredictorWithFeatureWeights<float>>> model)
        {
            using (var fs = new FileStream(_modelpath, FileMode.Create, FileAccess.Write, FileShare.Write))
                model.SaveTo(env, fs);

            Console.WriteLine("The model is saved to {0}", _modelpath);
        }

        private static IDataView GetData(LocalEnvironment env, string dataPath)
        {
             var reader = new TextLoader(env,
                            new TextLoader.Arguments()
                            {
                                Separator = "tab",
                                HasHeader = true,
                                Column = new[]
                                {
                                    new TextLoader.Column("Label", DataKind.Bool, 0),
                                    new TextLoader.Column("Text", DataKind.Text, 1)
                                }
                            });

                //Load training data 😒😒😒
            IDataView trainingDataView = reader.Read(new MultiFileSource(dataPath));
            return trainingDataView;
        }
        public static void PredictOne(string input) {
            var sentiment = new SentimentData { Text = input };
            
             using (var env2 = new LocalEnvironment())
            {
                ITransformer loadedModel;
                using (var stream = new FileStream(_modelpath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    loadedModel = TransformerChain.LoadFrom(env2, stream);
                }

                // Create prediction engine and make prediction.

                var engine = loadedModel.MakePredictionFunction<SentimentData, SentimentPrediction>(env2);

                var predictionFromLoaded = engine.Predict(sentiment);

                Console.WriteLine();
                Console.WriteLine("=============== Test of model with a sample ===============");

                Console.WriteLine($"Text: {sentiment.Text} | Prediction: {(Convert.ToBoolean(predictionFromLoaded.Prediction) ? "Toxic" : "Nice")} sentiment | Probability: {predictionFromLoaded.Probability} ");

            }
        }
    }
}
