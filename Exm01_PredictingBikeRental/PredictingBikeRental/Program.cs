using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictingBikeRental.DataProcessing;
using PredictingBikeRental.Models;
using PredictingBikeRental.Training;

namespace PredictingBikeRental
{
    class Program
    {
        // Путь к файлам данных
        private static string _dataPath = "../../Data/bike_sharing.csv";
        static void Main(string[] args)
        {
            Console.WriteLine("Предсказание типа аренды велосипеда с использованием машинного обучения");
            // 1. Создание ML.NET контекста
            var mlContext = new MLContext( seed: 0 );
            try
            {
            // 2. Загрузка данных
                Console.WriteLine("2. Загрузка данных...");
                Console.WriteLine("------");
                var dataProcessor = new DataProcessor( mlContext );
                var data = dataProcessor.LoadData( _dataPath );

             //3.Разделение данных на обучающую и тестовую выборки
                var sampleData = mlContext.Data.TakeRows( data, 1000 );
                dataProcessor.ExploreData( sampleData );


             // 4. Создание пайплайна обработки данных
                Console.WriteLine("4. Создание пайплайна обработки данных...");
                Console.WriteLine("------");
                var trainTestData = dataProcessor.SplitData( sampleData );
                var dataPrepPipeline = dataProcessor.CreateDataProcessingPipeline();

             // 5. Обучение моделей и выбор лучшей

             // 6. Оценка качества модели
                Console.WriteLine("3. Разделение данных...");
                Console.WriteLine("        * Разделение данных на обучающую и тестовую выборки...");
                Console.WriteLine("------");
                var modelTrainer = new ModelTrainer( mlContext );
             //var model = modelTrainer.TrainAndCompareModels( dataPrepPipeline, trainTestData.TrainSet );
                var bestModel = modelTrainer.TrainMultipleModels( dataPrepPipeline, trainTestData.TrainSet, trainTestData.TestSet );
                
                Console.WriteLine("\nОбучение успешно завершено!\n");
                // 7. Выполнение предсказаний
                //--. 4. Составление прогноза
                Console.WriteLine("4. Составление прогноза...");
                Console.WriteLine("------");
                var predictor = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);

                //Подготавливаем данные, по которым мы получим прогноз

                var sample = new BikeRentalData
                {
                    Season = 4,
                    Month = 8,
                    Hour = 10,
                    Holiday = 0,
                    Weekday = 5,
                    WorkingDay = 1,
                    WeatherCondition = 1,
                    Temperature = 31,
                    Humidity = 45,
                    Windspeed = 1
                };
                Console.WriteLine("Данные: " + sample.ToString() + "\n");
                //4.3.Отправляем наши данные в модель
                var prediction = predictor.Predict( sample );
                //4.4. Отображение прогноза
                Console.WriteLine($"\t*Rental type: {(prediction.PredictedRentalType ? "long-term" : "short-term")}, " +
                                    $"probability = {prediction.Probability:P1}");
                //--. 5. Сохрание модели для последующего использования, не тратя время на обучение
                Console.WriteLine("\nСохранение модели для последующего использования, не тратя время на обучение...");
                Console.WriteLine("------");
                mlContext.Model.Save( bestModel, trainTestData.TrainSet.Schema, "BikeRentalModel.zip" );
                Console.WriteLine("\t*Сохранение модели завершено успешно!\n");
            }
            catch ( Exception ex )
            {
                Console.WriteLine($"\nError: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
            Console.WriteLine("\nPress any key to complete...");
            Console.ReadKey();
        }
    }
}

