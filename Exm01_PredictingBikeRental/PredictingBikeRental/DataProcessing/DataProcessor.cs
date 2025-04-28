using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using PredictingBikeRental.Models;

namespace PredictingBikeRental.DataProcessing
{
    public class DataProcessor
    {
        private readonly MLContext _mlContext;

        //--.
        public DataProcessor( MLContext mlContext )
        {
            _mlContext = mlContext;
        }

        // --. Loads data from a CSV file
        public IDataView LoadData( string dataPath )
        {
            //--. Loading data taking into account format features
            var data = _mlContext.Data.LoadFromTextFile<BikeRentalData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                trimWhitespace: true);
            //--.
            Console.WriteLine($"\t*Data loaded from file: {dataPath}");
            return data;
        }


        // --. Performs exploratory data analysis
        public void ExploreData( IDataView data )
        {
            //--. Converting IDataView to a Collection for Analysis
            var bRList = _mlContext.Data
                .CreateEnumerable<BikeRentalData>( data, reuseRowObject: false )
                .ToList();

            Console.WriteLine($"\t*Number of entries: {bRList.Count}");

            
            //--. Analysis of numerical features
            Console.WriteLine("\n\t*Analysis of numerical features:");
            Console.WriteLine($"\t Temperature: Min={bRList.Min(x => x.Temperature)}, Max={bRList.Max(x => x.Temperature)}, Avg={bRList.Average(x => x.Temperature):F2}");
            Console.WriteLine($"\t Humidity: Min={bRList.Min(x => x.Humidity)}, Max={bRList.Max(x => x.Humidity)}, Avg={bRList.Average(x => x.Humidity):F2}");
            Console.WriteLine($"\t Windspeed: Min={bRList.Min(x => x.Windspeed)}, Max={bRList.Max(x => x.Windspeed)}, Avg={bRList.Average(x => x.Windspeed):F2}");
            

            // --. Analysis of the distribution of the target variable
            var incomeGroups = bRList.GroupBy(x => x.RentalType).Select(g => new { RentalType = g.Key, Count = g.Count() });
            Console.WriteLine("\n\t*Distribution of the target variable RentalType:");            
            foreach( var group in incomeGroups )
            {
                Console.WriteLine($"\t  {group.RentalType}: {group.Count} ({(float)group.Count / bRList.Count * 100:F2}%)");
            }
        }

        
        // --. Splits data into training and testing sets
        public Models.TrainTestData SplitData( IDataView data )
        {
            var mlSplitData = _mlContext.Data.TrainTestSplit( data, testFraction: 0.2 );
            var trainTestData = new Models.TrainTestData( mlSplitData.TrainSet, mlSplitData.TestSet );
            Console.WriteLine("\t*The data is divided into training and testing samples (80% / 20%)");
            
            return trainTestData;
        }


        public IEstimator<ITransformer> CreateDataProcessingPipeline()
        {
            Console.WriteLine("\t*Creating a data processing pipeline for bike rental...");

            //--. Создание конвейера обработки данных
            var dataPrepPipeline = _mlContext.Transforms.CustomMapping<BikeRentalData, BikeRentalWithBoolLabel>(
                    (input, output) =>
                    {
                        //--. Преобразование целевой переменной rental_type в булев тип
                        output.Label = input.RentalType; // Предполагается, что RentalType уже является 0 или 1
                    },
                    "RentalTypeMapping")

                // Нормализация числовых признаков
                .Append(_mlContext.Transforms.NormalizeMinMax("Temperature"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Humidity"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Windspeed"))

                // Кодирование категориальных признаков
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("SeasonEncoded", "Season"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("MonthEncoded", "Month"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("HourEncoded", "Hour"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("WeatherConditionEncoded", "WeatherCondition"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("WeekdayEncoded", "Weekday"))

                // Объединение всех признаков в один вектор
                .Append(_mlContext.Transforms.Concatenate("Features",
                    "SeasonEncoded", "MonthEncoded", "HourEncoded",
                    "Holiday", "WeekdayEncoded", "WorkingDay",
                    "WeatherConditionEncoded", "Temperature",
                    "Humidity", "Windspeed"));

            Console.WriteLine("\n\tA data processing pipeline for bike rental has been created");

            return dataPrepPipeline;
        }        
    }
}