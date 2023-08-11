import axios from 'axios';
import Papa from 'papaparse';

const BASE_URL = 'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/';

const TRAIN_FEATURES_FN = 'train-data.csv';
const TRAIN_TARGET_FN = 'train-target.csv';
const TEST_FEATURES_FN = 'test-data.csv';
const TEST_TARGET_FN = 'test-target.csv';

const parseCsv = async (data: any) => {
  return new Promise((resolve) => {
    data = data.map((row: any) => {
      return Object.keys(row).map((key) => parseFloat(row[key]));
    });
    resolve(data);
  });
};

const shuffle = (data: any, target: any) => {
  let counter = data.length;
  let temp = 0;
  let index = 0;
  while (counter > 0) {
    index = (Math.random() * counter) | 0;
    counter--;
    // 데이터:
    temp = data[counter];
    data[counter] = data[index];
    data[index] = temp;
    // 타깃:
    temp = target[counter];
    target[counter] = target[index];
    target[index] = temp;
  }
};

export const loadCsv = async (filename: string) => {
  try {
    const url = `${BASE_URL}${filename}`;
    const loadData = await axios.get(url, { responseType: 'blob' });
    const csvData = Papa.parse(loadData.data, {
      header: true,
    });
    const parseCsvData = await parseCsv(csvData['data']);
    return parseCsvData;
  } catch (error) {
    console.log('loadCsv error: ', error);
  }
};

export type BostonHousingDataSet = Number[][];

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
    const [trainFeatures, trainTarget, testFeatures, testTarget] = await axios.all([
      loadCsv(TRAIN_FEATURES_FN),
      loadCsv(TRAIN_TARGET_FN),
      loadCsv(TEST_FEATURES_FN),
      loadCsv(TEST_TARGET_FN),
    ]);

    this.trainFeatures = trainFeatures as BostonHousingDataSet;
    this.trainTarget = trainTarget as BostonHousingDataSet;
    this.testFeatures = testFeatures as BostonHousingDataSet;
    this.testTarget = testTarget as BostonHousingDataSet;

    shuffle(this.trainFeatures, this.trainTarget);
    shuffle(this.testFeatures, this.testTarget);
  }
}
