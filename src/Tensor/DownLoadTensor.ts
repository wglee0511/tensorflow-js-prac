import { tf } from '@/constant/globalTf';
import { DOWNLOAD_TEST_DATA, DOWNLOAD_TRAIN_DATA } from '@/data/DowmLoadData';

export const downLoadTrainTensors = {
  sizeMB: tf.tensor2d(DOWNLOAD_TRAIN_DATA.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(DOWNLOAD_TRAIN_DATA.timeSec, [20, 1]),
};

export const downLoadTestTensors = {
  sizeMB: tf.tensor2d(DOWNLOAD_TEST_DATA.sizeMB, [20, 1]),
  timeSec: tf.tensor2d(DOWNLOAD_TEST_DATA.timeSec, [20, 1]),
};
