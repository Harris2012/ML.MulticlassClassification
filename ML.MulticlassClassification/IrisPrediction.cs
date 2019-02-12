using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.MulticlassClassification
{
    /// <summary>
    /// IrisPrediction is the result returned from prediction operations
    /// </summary>
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
