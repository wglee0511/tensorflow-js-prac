import { tf } from '@/constant/globalTf';
import { loadCsv } from '@/util/loadData';
import { normalizeDataset } from '@/util/normalization';
import axios from 'axios';
import { flatten } from 'lodash';

const TRAIN_DATA = 'train-data.csv';
const TRAIN_TARGET = 'train-target.csv';
const TEST_DATA = 'test-data.csv';
const TEST_TARGET = 'test-target.csv';

const BASE_URL =
  'https://gist.githubusercontent.com/ManrajGrover/6589d3fd3eb9a0719d2a83128741dfc1/raw/d0a86602a87bfe147c240e87e6a9641786cafc19/';

type WebsitePhishingRawDataSet = number[][];

export class WebsitePhishingDataSet {
  public trainFeatures: WebsitePhishingRawDataSet | null = null;
  public trainTarget: WebsitePhishingRawDataSet | null = null;
  public testFeatures: WebsitePhishingRawDataSet | null = null;
  public testTarget: WebsitePhishingRawDataSet | null = null;
  public trainSize: number = 0;
  public testSize: number = 0;
  public trainBatchIndex: number = 0;
  public testBatchIndex: number = 0;
  public numFeatures: number = 0;
  public NUM_CLASSES: number = 2;

  async loadData() {
    const [trainFeatures, trainTarget, testFeatures, testTarget] =
      await axios.all([
        loadCsv(BASE_URL, TRAIN_DATA),
        loadCsv(BASE_URL, TRAIN_TARGET),
        loadCsv(BASE_URL, TEST_DATA),
        loadCsv(BASE_URL, TEST_TARGET),
      ]);

    const {
      data: normalizedTrainFeatures,
      vectorMeans,
      vectorStddevs,
    } = normalizeDataset({
      data: trainFeatures as WebsitePhishingRawDataSet,
    });

    const { data: normalizedTrainTargets } = normalizeDataset({
      data: trainTarget as WebsitePhishingRawDataSet,
      isTrainData: false,
      vectorMeans,
      vectorStddevs,
    });

    this.trainFeatures = normalizedTrainFeatures;
    this.trainTarget = normalizedTrainTargets;
    this.testFeatures = testFeatures as WebsitePhishingRawDataSet;
    this.testTarget = testTarget as WebsitePhishingRawDataSet;
    this.trainSize = (trainFeatures as WebsitePhishingRawDataSet)?.length || 0;
    this.testSize = (testFeatures as WebsitePhishingRawDataSet)?.length || 0;
    this.numFeatures = normalizedTrainFeatures[0]?.length;
  }

  getTrainData() {
    const dataShape: [number, number] = [this.trainSize, this.numFeatures];
    const trainData = Float32Array.from(flatten(this.trainFeatures));
    const trainTarget = Float32Array.from(flatten(this.trainTarget));
    return {
      data: tf.tensor2d(trainData, dataShape),
      target: tf.tensor1d(trainTarget),
    };
  }

  getTestData() {
    const dataShape: [number, number] = [this.testSize, this.numFeatures];
    const testData = Float32Array.from(flatten(this.testFeatures));
    const testTarget = Float32Array.from(flatten(this.testTarget));

    return {
      data: tf.tensor2d(testData, dataShape),
      target: tf.tensor1d(testTarget),
    };
  }
}
