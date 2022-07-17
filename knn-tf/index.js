require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
  const unSortedFeatures = features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1);
  // start of code to sort features by first column
  const firstAxis = unSortedFeatures.gather([0], 1).reshape([-1]);
  const ind = tf.topk(firstAxis, unSortedFeatures.shape[0]).indices;
  return (
    unSortedFeatures
      .gather(ind.reverse(), 0)
      // end of code to sort features by first column
      .unstack()
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = ((testLabels[i][0] - result) / testLabels[i][0]) * 100;
  console.log("Error is ", err, "%");
});
