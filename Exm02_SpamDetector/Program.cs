using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SpamDetector
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //Load sample data
            var sampleData = new MLModel1.ModelInput()
            {
                Content = @"Hi team, Just a reminder that we have our weekly project status meeting tomorrow at 10 AM in Conference Room B. Please bring your status reports and be prepared to discuss the timeline updates. Thanks, Sarah",
            };
            //Load model and predict output
            var result = MLModel1.Predict(sampleData);
            Console.WriteLine(result.PredictedLabel);
        }
    }
}
