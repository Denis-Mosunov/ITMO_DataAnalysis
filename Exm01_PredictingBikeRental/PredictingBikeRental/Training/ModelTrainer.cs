using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictingBikeRental.Models;
using Microsoft.ML.Trainers;


namespace PredictingBikeRental.Training
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;
        
        public ModelTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }
        public ITransformer TrainAndCompareModels(IEstimator<ITransformer> dataPrepPipeline, IDataView trainData)
        {
            Console.WriteLine("\t*Training a simplified model:");

            //--. We use the simplest call without additional parameters
            var estimator = _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
            
            Console.WriteLine("\t*Building a pipeline...");

            //--. Creating a complete pipeline (preprocessing + training)
            var pipeline = dataPrepPipeline.Append( estimator );

            //--. Training a model with time dimension
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            
            Console.WriteLine("\t*Start of Fit() operation...");
            var model = pipeline.Fit(trainData);
            stopWatch.Stop();
            
            Console.WriteLine("\n\tThe Fit() operation has completed.");
            Console.WriteLine($"Training time: {stopWatch.ElapsedMilliseconds / 1000.0:F2} seconds");

            //--. Performing a simple assessment
            Console.WriteLine("Performing Model Evaluation...");
            var predictions = model.Transform( trainData );
            var metrics = _mlContext.BinaryClassification.Evaluate( predictions );
            
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
            return model;
        }
        public ITransformer TrainMultipleModels(IEstimator<ITransformer> dataPrepPipeline, IDataView trainData, IDataView testData)
        {
            var models = new List<(string Name, ITransformer Model, double Accuracy)>();

            //--. List of algorithms for training
            var trainers = new List<(string Name, IEstimator<ITransformer> Estimator)>
            {
                ("SDCA Logistic Regression", _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression()),
                ("FastTree", _mlContext.BinaryClassification.Trainers.FastTree()),
                ("LightGBM", _mlContext.BinaryClassification.Trainers.LightGbm())
            };

            //--.
            foreach( var trainer in trainers )
            {
                Console.WriteLine($"\t*Training model: {trainer.Name}...");
                var pipeline = dataPrepPipeline.Append(trainer.Estimator);

                var model = pipeline.Fit(trainData);
                var predictions = model.Transform(testData);        //<-- test, not train data!!!
                var metrics = _mlContext.BinaryClassification.Evaluate(predictions);

                models.Add((trainer.Name, model, metrics.Accuracy));
                Console.WriteLine($"\t*Model: {trainer.Name}, Accuracy: {metrics.Accuracy:F4}");
            }

            //--. Choosing the best model
            var bestModel = models.OrderByDescending(m => m.Accuracy).First();
            Console.WriteLine($"\t**Best Model: {bestModel.Name} with Accuracy: {bestModel.Accuracy:F4}");

            return bestModel.Model;
        }
    }
}