import { tf } from '@/constant/globalTf';

export const getWebsitePhishingModel = ({
  numFeatures,
}: {
  numFeatures: number;
}) => {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [numFeatures],
      units: 100,
      activation: 'sigmoid',
    }),
  );

  model.add(
    tf.layers.dense({
      units: 100,
      activation: 'sigmoid',
    }),
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
    }),
  );

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};
