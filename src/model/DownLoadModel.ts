import { tf } from '@/constant/globalTf';

const model = tf.sequential();

model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
model.compile({ optimizer: 'sgd', loss: 'meanAbsoluteError' });

export { model as DownloadModel };
