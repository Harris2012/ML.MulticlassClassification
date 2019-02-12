using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.MulticlassClassification
{
    /// <summary>
    /// IrisData is used to provide training data, and as input for prediction operations.
    /// - First 4 properties are inputs/features used to predict the label
    /// - Label is what you are predicting, and is only set when training
    /// </summary>
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string Label;
    }
}
