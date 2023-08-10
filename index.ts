import { BostonHousingPriceDataSet } from '@/data/BostonHousingPrice';
import { downloadTrain } from './src';

//downloadTrain();
const bostonDataset = new BostonHousingPriceDataSet();
bostonDataset.loadData();

export {};
