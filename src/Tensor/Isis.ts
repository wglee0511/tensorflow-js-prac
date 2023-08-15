import { tf } from '@/constant/globalTf';
import { IRIS_NUM_CLASSES } from '@/data/Iris';
import { shuffle } from 'lodash';

export const getIrisTensors = ({
  dataByClass,
  targetByClass,
  testSplit,
}: {
  dataByClass: number[][];
  targetByClass: number[];
  testSplit: number;
}) => {
  const numExamples = dataByClass.length;
  if (numExamples !== targetByClass.length) {
    throw new Error('데이터와 타깃의 길이가 다릅니다.');
  }

  const shuffledData = shuffle(dataByClass);
  const shuffledTarget = shuffle(targetByClass);

  // `testSplit`를 기준으로 데이터를 훈련 세트와 테스트 세트로 나눕니다.
  const numTestExamples = Math.round(numExamples * testSplit);
  const numTrainExamples = numExamples - numTestExamples;

  const xDims = shuffledData[0].length;

  const xs = tf.tensor2d(shuffledData, [numExamples, xDims]);
  const ys = tf.oneHot(tf.tensor1d(shuffledTarget).toInt(), IRIS_NUM_CLASSES);

  const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
  const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
  const yTrain = ys.slice([0, 0], [numTrainExamples, IRIS_NUM_CLASSES]);
  const yTest = ys.slice([0, 0], [numTestExamples, IRIS_NUM_CLASSES]);

  return { xTrain, xTest, yTrain, yTest };
};
