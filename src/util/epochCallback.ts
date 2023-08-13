import { Logs } from '@tensorflow/tfjs';

export const onEpochEnd = async ({
  epoch,
  logs,
  numEpochs,
  trainLogs,
}: {
  epoch: number;
  logs: Logs | undefined;
  numEpochs: number;
  trainLogs: Logs[];
}) => {
  console.log('logs: ', logs);
  trainLogs.push(logs || {});
  console.log(`에포크 ${numEpochs}번 중 ${epoch + 1}번째 완료`);
};

export const onEpochBegin = async (epoch: number) => {
  console.log(epoch);
};
