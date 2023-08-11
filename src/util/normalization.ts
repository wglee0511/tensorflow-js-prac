import { tf } from '@/constant/globalTf';

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
