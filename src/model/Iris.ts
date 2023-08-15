import { tf } from '@/constant/globalTf';
import { getIrisData } from '@/data/Iris';
import { layers } from '@tensorflow/tfjs';

export const getIrisModel = (testSplit: number) => {
  const { tfXTrains } = getIrisData(testSplit);

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
