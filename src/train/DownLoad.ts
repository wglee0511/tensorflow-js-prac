import { downLoadTestTensors, downLoadTrainTensors } from '@/Tensor/DownLoadTensor';
import { tf } from '@/constant/globalTf';
import { DOWNLOAD_TEST_DATA, DOWNLOAD_TRAIN_DATA } from '@/data/DowmLoadData';
import { DownloadModel } from '@/model/DownLoadModel';
import { Tensor, Tensor1D, Tensor2D } from '@tensorflow/tfjs';

export const downloadTrain = async () => {
  await DownloadModel.fit(downLoadTrainTensors.sizeMB, downLoadTrainTensors.timeSec, {
    epochs: 1000,
  });
  const evaluateModel = DownloadModel.evaluate(downLoadTestTensors.sizeMB, downLoadTestTensors.timeSec) as Tensor;
  evaluateModel.print();
  tf.mean(DOWNLOAD_TRAIN_DATA.timeSec).print();
  tf.mean(tf.abs(tf.sub(DOWNLOAD_TEST_DATA.timeSec, 0.295))).print();

  const smallFileMB = 1;
  const bigFileMB = 100;
  const hugeFileMB = 10000;

  const predictedDownLoadValueTensor2d = DownloadModel.predict(
    tf.tensor2d([[smallFileMB], [bigFileMB], [hugeFileMB]]),
  ) as Tensor2D;

  predictedDownLoadValueTensor2d.print();

  const predictedDownLoadValueTensor1d = DownloadModel.predict(
    tf.tensor1d([smallFileMB, bigFileMB, hugeFileMB]),
  ) as Tensor1D;

  predictedDownLoadValueTensor1d.print();
};
