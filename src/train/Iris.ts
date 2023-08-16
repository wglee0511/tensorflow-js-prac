import { tf } from '@/constant/globalTf';
import { HOSTED_MODEL_JSON_URL, IRIS_CLASSES, getIrisData } from '@/data/Iris';
import { getIrisModel } from '@/model/Iris';
import { Tensor } from '@tensorflow/tfjs';

export const irisTrain = async () => {
  // 외부 입력사항
  const inputData = [5.1, 3.5, 1.4, 0.2];
  const testSplit = 0.15;
  const params = {
    epochs: 40,
    learningRate: 0.01,
  };

  const model = getIrisModel(testSplit);
  const { tfXTrains, tfYTrains, tfXTests, tfYTests } = getIrisData(testSplit);

  console.log('모델 훈현중  ==================================');

  await model.fit(tfXTrains, tfYTrains, {
    epochs: params.epochs,
    validationData: [tfXTests, tfYTests],
  });

  const predictResult = tf.tidy(() => {
    const inputTensorData = tf.tensor2d([inputData], [1, 4]);
    const predictTensorData = model.predict(inputTensorData) as Tensor;
    const winnerValueIndex = predictTensorData.argMax(-1).dataSync()[0];

    return IRIS_CLASSES[winnerValueIndex];
  });

  console.log('입력값에 대한 예측 종류 🌸 : ', predictResult);

  const trainedModel = await tf.loadLayersModel(HOSTED_MODEL_JSON_URL);

  const trainedResult = tf.tidy(() => {
    const inputTensorData = tf.tensor2d([inputData], [1, 4]);
    const predictTensorData = trainedModel.predict(inputTensorData) as Tensor;
    const winnerValueIndex = predictTensorData.argMax(-1).dataSync()[0];

    return IRIS_CLASSES[winnerValueIndex];
  });
  console.log('훈련된 값에 대한 예측 종류 🍯 : ', trainedResult);
};
