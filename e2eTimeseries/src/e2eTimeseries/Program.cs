using System;
using System.Collections.Generic;

using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.TimeSeriesProcessing;

namespace e2eTimeseries
{
    class Program
    {
        static void Main(string[] args)  
        {
            using (var env = new ConsoleEnvironment(conc: 1))
            {
                const int ChangeHistorySize = 2000;
                const int SeasonalitySize = 1000;
                const int NumberOfSeasonsInTraining = 5;
                const int MaxTrainingSize = NumberOfSeasonsInTraining * SeasonalitySize;

                List<Data> data = new List<Data>();
                var dataView = env.CreateStreamingDataView(data);

                var args = new SsaChangePointDetector.Arguments()
                {
                    Confidence = 95,
                    Source = "Value",
                    Name = "Change",
                    ChangeHistoryLength = ChangeHistorySize,
                    Data = dataView,
                    TrainingWindowSize = MaxTrainingSize,
                    SeasonalWindowSize = SeasonalitySize
                };

                for (int j = 0; j < NumberOfSeasonsInTraining; j++)
                    for (int i = 0; i < SeasonalitySize; i++)
                        data.Add(new Data(i));

                for (int i = 0; i < ChangeHistorySize; i++)
                    data.Add(new Data(i * 100));

                var detector = TimeSeriesProcessing.SsaChangePointDetector(env, args);
                var output = detector.Model.Apply(env, dataView);
                var enumerator = output.AsEnumerable<Prediction>(env, true).GetEnumerator();
                Prediction row = null;
                List<double> expectedValues = new List<double>() { 0, 0, 0.5, 0, 0, 1, 0.15865526383236372,
                    0, 0, 1.6069464981555939, 0.05652458872960725, 0, 0, 2.0183047652244568, 0.11021633531076747, 0};

                int index = 0;
                while (enumerator.MoveNext() && index < expectedValues.Count)
                {
                    row = enumerator.Current;
                    Console.WriteLine($"expectedValues[index++], row.Change[0]");
                    Console.WriteLine($"expectedValues[index++], row.Change[1]");
                    Console.WriteLine($"expectedValues[index++], row.Change[2]");
                    Console.WriteLine($"expectedValues[index++], row.Change[3]");
                }
            }
        }
    }
}
