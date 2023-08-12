import { loadCsv, shuffle } from '@/util/loadData';
import axios from 'axios';

const BASE_URL =
  'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

const TRAIN_FEATURES_FN = 'train-data.csv';
const TRAIN_TARGET_FN = 'train-target.csv';
const TEST_FEATURES_FN = 'test-data.csv';
const TEST_TARGET_FN = 'test-target.csv';

export type BostonHousingDataSet = number[][];

export class BostonHousingPriceDataSet {
  public trainFeatures: BostonHousingDataSet | null = null;
  public trainTarget: BostonHousingDataSet | null = null;
  public testFeatures: BostonHousingDataSet | null = null;
  public testTarget: BostonHousingDataSet | null = null;

  get numFeatures() {
    // 데이터를 로드하기 전에 numFetures를 참조하면 에러를 발생시킵니다.
    if (this.trainFeatures == null) {
      throw new Error("numFeatures 전에 'loadData()'를 호출해야 합니다.");
    }
    return this.trainFeatures[0].length;
  }

  async loadData() {
    const [trainFeatures, trainTarget, testFeatures, testTarget] =
      await axios.all([
        loadCsv(BASE_URL, TRAIN_FEATURES_FN),
        loadCsv(BASE_URL, TRAIN_TARGET_FN),
        loadCsv(BASE_URL, TEST_FEATURES_FN),
        loadCsv(BASE_URL, TEST_TARGET_FN),
      ]);

    this.trainFeatures = trainFeatures as BostonHousingDataSet;
    this.trainTarget = trainTarget as BostonHousingDataSet;
    this.testFeatures = testFeatures as BostonHousingDataSet;
    this.testTarget = testTarget as BostonHousingDataSet;

    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);
  }
}
