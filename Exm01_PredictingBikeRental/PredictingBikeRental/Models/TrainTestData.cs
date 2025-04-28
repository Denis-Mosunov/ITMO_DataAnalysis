using Microsoft.ML;
using Microsoft.ML.Data;

namespace PredictingBikeRental.Models
{
    //--. Class for storing separated training and test samples
    public class TrainTestData
    {
        public IDataView TrainSet { get; set; }
        public IDataView TestSet { get; set; }
        
        public TrainTestData(IDataView trainSet, IDataView testSet)
        {
            TrainSet = trainSet;
            TestSet = testSet;
        }
    }
}