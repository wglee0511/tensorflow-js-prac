import { downLoadTrainTensors } from '@/Tensor/DownLoadTensor';
import { DownloadModel } from '@/model/DownLoadModel';

export const downloadTrain = async () => {
  await DownloadModel.fit(downLoadTrainTensors.sizeMB, downLoadTrainTensors.timeSec, { epochs: 10 });
};
