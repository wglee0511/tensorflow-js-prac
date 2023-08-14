import { tf } from '@/constant/globalTf';
import { getIrisData } from '@/data/Iris';
import { Tensor, layers } from '@tensorflow/tfjs';

export const getIrisModel = () => {
  const { tfYTests, tfXTests, tfXTrains, tfYTrains } = getIrisData(0.15);

  const model = tf.sequential();

  model.add(
    layers.dense({
      inputShape: [tfXTrains.shape[1]] as number[],
      units: 10,
      activation: 'sigmoid',
    }),
  );

  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

  model.summary();

  const LEARNING_RATE = 0.01;
  const optimizer = tf.train.adam(LEARNING_RATE);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};
