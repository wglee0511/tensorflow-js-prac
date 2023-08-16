import { tf } from '@/constant/globalTf';
import { determineMeanAndStddev, normalizeTensors } from '@/util/normalization';
import { TensorLike2D } from '@tensorflow/tfjs-core/dist/types';

export const bostonHousingPriceTensors = ({
  dataSetTrainFeatures,
  dataSetTrainTarget,
  dataSetTestFeatures,
  dataSetTestTarget,
}: {
  dataSetTrainFeatures: TensorLike2D;
  dataSetTrainTarget: TensorLike2D;
  dataSetTestFeatures: TensorLike2D;
  dataSetTestTarget: TensorLike2D;
}) => {
  const rawTrainFeatures = tf.tensor2d(dataSetTrainFeatures);
  const trainTarget = tf.tensor2d(dataSetTrainTarget);
  const rawTestFeatures = tf.tensor2d(dataSetTestFeatures);
  const testTarget = tf.tensor2d(dataSetTestTarget);

  const { dataMean: trainDataMean, dataStd: trainDataStd } = determineMeanAndStddev(rawTrainFeatures);
  const { dataMean: testDataMean, dataStd: testDataStd } = determineMeanAndStddev(rawTestFeatures);
  const trainFeatures = normalizeTensors(rawTrainFeatures, trainDataMean, trainDataStd);
  const testFeatures = normalizeTensors(rawTestFeatures, testDataMean, testDataStd);

  return {
    trainFeatures,
    trainTarget,
    testFeatures,
    testTarget,
  };
};
