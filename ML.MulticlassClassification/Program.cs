using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace ML.MulticlassClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            //训练模型
            TrainAndSave(context);

            //预测结果
            Predicate(context);

            Console.Read();
        }

        private static void TrainAndSave(MLContext mlContext)
        {
            var reader = mlContext.Data.CreateTextReader<IrisData>(separatorChar: ',', hasHeader: false);
            IDataView trainingDataView = reader.Read("iris.data.txt");

            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = mlContext.Transforms.Conversion
                .MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // STEP 4: Train your model based on the data set  
            var model = pipeline.Fit(trainingDataView);
            using (var stream = new FileStream("the-model.zip", FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, stream);
            }
        }

        private static void Predicate(MLContext mlContext)
        {
            ITransformer transformer;
            using (var stream = new FileStream("the-model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                transformer = mlContext.Model.Load(stream);
            }

            var prediction = transformer.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = 3.3f,
                    SepalWidth = 1.6f,
                    PetalLength = 0.2f,
                    PetalWidth = 5.1f,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }
    }
}
