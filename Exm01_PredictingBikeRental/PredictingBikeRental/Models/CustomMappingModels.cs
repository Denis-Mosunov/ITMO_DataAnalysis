using Microsoft.ML.Data;

namespace PredictingBikeRental.Models
{

    /*
    /// Вспомогательный класс для трансформации капитала (обработка выбросов)
    public class CapitalTransformedData
    {
        public float LogCapitalGain { get; set; }
        public float LogCapitalLoss { get; set; }
    }

    /// Вспомогательный класс для порядкового кодирования образования
    public class EducationMappingData
    {
        public float EducationOrdinal { get; set; }
    }

    /// Вспомогательный класс для порядкового кодирования семейного положения
    public class MaritalMappingData
    {
        public float MaritalOrdinal { get; set; }
    }

    /// Вспомогательный класс для создания производных признаков
    public class DerivedFeaturesData
    {
        public float HasCapitalIncome { get; set; }
        public float AgeGroup { get; set; }
        public float WorkHoursCategory { get; set; }
    }

    /// Класс для представления преобразованного набора данных
    public class TransformedAdultData
    {
        public float Age { get; set; }
        public float EducationNum { get; set; }
        public float HoursPerWeek { get; set; }
        public float LogCapitalGain { get; set; }
        public float LogCapitalLoss { get; set; }
        public float[]? WorkclassEncoded { get; set; }  // Добавляем ? для nullable
        public float EducationOrdinal { get; set; }
        public float MaritalOrdinal { get; set; }
        public float HasCapitalIncome { get; set; }
        public float AgeGroup { get; set; }
        public float WorkHoursCategory { get; set; }
        public float[]? Features { get; set; }  // Добавляем ? для nullable
        public bool Label { get; set; }
    }
    */

    //--. Класс для преобразования строкового RentalType в булево значение
    public class BikeRentalWithBoolLabel
    {
        [ColumnName("Label")]
        public bool Label { get; set; }
    }
}