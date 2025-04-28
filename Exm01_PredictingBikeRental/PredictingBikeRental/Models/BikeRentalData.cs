using Microsoft.ML.Data;

namespace PredictingBikeRental.Models
{
    public class BikeRentalData
    {
        [LoadColumn(0)]
        public float Season { get; set; }

        [LoadColumn(1)]
        public float Month { get; set; }

        [LoadColumn(2)]
        public float Hour { get; set; }

        [LoadColumn(3)]
        public float Holiday { get; set; }

        [LoadColumn(4)]
        public float Weekday { get; set; }

        [LoadColumn(5)]
        public float WorkingDay { get; set; }

        [LoadColumn(6)]
        public float WeatherCondition { get; set; }

        [LoadColumn(7)]
        public float Temperature { get; set; }

        [LoadColumn(8)]
        public float Humidity { get; set; }

        [LoadColumn(9)]
        public float Windspeed { get; set; }

        [LoadColumn(10)]
        public bool RentalType { get; set; } // 0 = short-term, 1 = long-term

        //--.
        public override string ToString()
        {
            return $"\tSeason: {Season}\n\t\t\tMonth: {Month}\n\t\t\tHour: {Hour}\n\t\t\tHoliday: {Holiday}\n\t\t\t" +
                   $"Weekday: {Weekday}\n\t\t\tWorkingDay: {WorkingDay}\n\t\t\tWeatherCondition: {WeatherCondition}\n\t\t\t" +
                   $"Temperature: {Temperature}°C\n\t\t\tHumidity: {Humidity}%\n\t\t\tWindspeed: {Windspeed} km/h";
        }
    }
}