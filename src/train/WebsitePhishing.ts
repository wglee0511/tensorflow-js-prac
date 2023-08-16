import { WebsitePhishingDataSet } from '@/data/WebsitePhishing';
import { getWebsitePhishingModel } from '@/model/websitePhishing';
import { onEpochEnd } from '@/util/epochCallback';
import { Logs, Tensor } from '@tensorflow/tfjs';

const EPOCHS = 400;
const BATCH_SIZES = 350;

export const websitePhishingTrain = async () => {
  const websitePhishingDataSet = new WebsitePhishingDataSet();
  await websitePhishingDataSet.loadData();

  const trainLogs: Logs[] = [];

  const { data: trainData, target: trainTarget } = websitePhishingDataSet.getTrainData();
  const { data: testData, target: testTarget } = websitePhishingDataSet.getTestData();

  const model = getWebsitePhishingModel({
    numFeatures: websitePhishingDataSet.numFeatures,
  });

  await model.fit(trainData, trainTarget, {
    batchSize: BATCH_SIZES,
    epochs: EPOCHS,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) =>
        onEpochEnd({
          epoch,
          logs,
          numEpochs: EPOCHS,
          trainLogs: trainLogs,
        }),
    },
  });

  const resultObj = model.evaluate(testData, testTarget, {
    batchSize: BATCH_SIZES,
  }) as Tensor[];

  const result = resultObj[0].dataSync()[0];

  console.log('최종 결과: ', result);
};
