import { getIrisTensors } from '@/Tensor/Isis';
import { getIrisData } from '@/data/Iris';
import { getIrisModel } from '@/model/Iris';

export const irisTrain = async () => {
  getIrisModel();
};
