import { LEARNING_RATE } from '@/constant/constant';
import { tf } from '@/constant/globalTf';
import { Logs } from '@tensorflow/tfjs';

export const getBostonHousingPriceModel = ({
  numFeatures = 0,
}: {
  numFeatures: number;
}) => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [numFeatures], units: 1 }));
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError',
  });

  return model;
};

export const getMultiLayerPerceptronRegressionModelHidden = ({
  numFeatures = 0,
}: {
  numFeatures: number;
}) => {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [numFeatures],
      units: 50,
      activation: 'sigmoid',
      kernelInitializer: 'leCunNormal',
    }),
  );

  // 은닉층 추가
  model.add(
    tf.layers.dense({
      units: 50,
      activation: 'sigmoid',
      kernelInitializer: 'leCunNormal',
    }),
  );
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError',
  });
  model.summary();

  return model;
};

export const getBostonHousingPriceModelCallbacks = () => {
  const onEpochEnd = async ({
    epoch,
    logs,
    numEpochs,
    trainLogs,
  }: {
    epoch: number;
    logs: Logs | undefined;
    numEpochs: number;
    trainLogs: Logs[];
  }) => {
    console.log('logs: ', logs);
    trainLogs.push(logs || {});
    console.log(`에포크 ${numEpochs}번 중 ${epoch + 1}번째 완료`);
  };

  return { onEpochEnd };
};
