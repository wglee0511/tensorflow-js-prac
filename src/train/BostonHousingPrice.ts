import { bostonHousingPriceTensors } from '@/Tensor/BostonHousingPrice';
import { BostonHousingPriceDataSet } from '@/data/BostonHousingPrice';
import {
  // getBostonHousingPriceModel,
  getBostonHousingPriceModelCallbacks,
  getMultiLayerPerceptronRegressionModelHidden,
} from '@/model/BostonHousingPrice';
import { computeBaseline } from '@/util/computeBaseline';
import { Logs, Tensor } from '@tensorflow/tfjs';
import { TensorLike2D } from '@tensorflow/tfjs-core/dist/types';

const NUM_EPOCHS = 200;
const BATCH_SIZE = 40;

export const bostonHousingTrain = async () => {
  const { onEpochEnd } = getBostonHousingPriceModelCallbacks();
  const bostonHousingDataSet = new BostonHousingPriceDataSet();
  const trainLogs: Logs[] = [];

  await bostonHousingDataSet.loadData();
  const bostonHousingModel = getMultiLayerPerceptronRegressionModelHidden({
    numFeatures: bostonHousingDataSet.numFeatures,
  });

  const { trainTarget, testTarget, testFeatures, trainFeatures } =
    bostonHousingPriceTensors({
      dataSetTestFeatures: bostonHousingDataSet.testFeatures as TensorLike2D,
      dataSetTestTarget: bostonHousingDataSet.testTarget as TensorLike2D,
      dataSetTrainFeatures: bostonHousingDataSet.trainFeatures as TensorLike2D,
      dataSetTrainTarget: bostonHousingDataSet.trainTarget as TensorLike2D,
    });

  await bostonHousingModel.fit(trainFeatures as Tensor, trainTarget as Tensor, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: async (epoch, logs) =>
        onEpochEnd({
          epoch,
          logs,
          numEpochs: NUM_EPOCHS,
          trainLogs: trainLogs,
        }),
    },
  });

  computeBaseline({
    trainTarget: trainTarget,
    testTarget: testTarget,
    testName: '보스턴 주택 가격',
  });

  console.log('평가 시작 ==================================');

  const bostonHousingPriceResult = bostonHousingModel.evaluate(
    testFeatures,
    testTarget,
    {
      batchSize: BATCH_SIZE,
    },
  ) as Tensor;

  const testLoss = bostonHousingPriceResult.dataSync()[0];
  const trainLoss = trainLogs[trainLogs.length - 1].loss;
  const valLoss = trainLogs[trainLogs.length - 1].val_loss;

  const resultTestLoss = Number(testLoss.toFixed(4));
  const trainTestLoss = Number(trainLoss.toFixed(4));
  const valTestLoss = Number(valLoss.toFixed(4));

  console.log('훈련 세트 최종 손실: ', trainLoss.toFixed(4));
  console.log('검증 세트 최종 손실: ', valLoss.toFixed(4));
  console.log('테스트 세트 최종 손실: ', testLoss.toFixed(4));
  console.log('훈련세트 제곱근: ', Math.sqrt(resultTestLoss));
  console.log('검증 세트 제곱근: ', Math.sqrt(trainTestLoss));
  console.log('테스트 세트 제곱근: ', Math.sqrt(valTestLoss));

  const weightValue = bostonHousingModel.layers[0].getWeights()[0].dataSync();
  console.log('가중치 데이터: ', weightValue);
};
