import { LEARNING_RATE } from '@/constant/constant';
import { tf } from '@/constant/globalTf';

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
