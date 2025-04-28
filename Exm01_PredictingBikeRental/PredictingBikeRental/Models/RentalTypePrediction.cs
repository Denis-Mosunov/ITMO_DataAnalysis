using Microsoft.ML.Data;

namespace PredictingBikeRental.Models
{
    //--. Класс для представления результатов предсказания
    public class RentalTypePrediction
    {
        //--. Предсказанное значение (тип аренды: 0=краткосрочная, 1=долгосрочная)
        [ColumnName("PredictedLabel")]
        public bool PredictedRentalType { get; set; }

        //--. Вероятность того, что доход 1-долгострочная
        public float Probability { get; set; }

        //--ю Значение оценки перед преобразованием в вероятность
        public float Score { get; set; }
    }
}
