import { tf } from '@/constant/globalTf';
import { map, meanBy } from 'lodash';
import { normalizeVector, stddev } from './calculate';
import { Vector } from './type';

export const determineMeanAndStddev = (data: tf.Tensor) => {
  const dataMean = data.mean(0);
  const differenceFromMean = data.sub(dataMean);
  const squaredDifferenceFromMean = differenceFromMean.square();
  const variance = squaredDifferenceFromMean.mean(0);
  const dataStd = variance.sqrt();

  return { dataMean, dataStd };
};

export const normalizeTensors = (data: tf.Tensor, dataMean: tf.Tensor, dataStd: tf.Tensor): tf.Tensor => {
  return data.sub(dataMean).div(dataStd);
};

export const normalizeDataset = ({
  data,
  isTrainData = true,
  vectorMeans = [],
  vectorStddevs = [],
}: {
  data: number[][];
  isTrainData?: boolean;
  vectorMeans?: Vector;
  vectorStddevs?: Vector;
}) => {
  const numFeatures = data[0].length;
  let vectorMean: number;
  let vectorStddev: number;

  for (let i = 0; i < numFeatures; i++) {
    const vector: Vector = map(data, (row) => {
      return row[i];
    });

    if (isTrainData) {
      vectorMean = meanBy(vector);
      vectorStddev = stddev(vector);

      vectorMeans.push(vectorMean);
      vectorStddevs.push(vectorStddev);
    } else {
      vectorMean = vectorMeans[i];
      vectorStddev = vectorStddevs[i];
    }

    const vectorNormalized = normalizeVector(vector, vectorMean, vectorStddev);

    vectorNormalized.forEach((value, index) => {
      data[index][i] = value;
    });
  }

  return { data, vectorMeans, vectorStddevs };
};
