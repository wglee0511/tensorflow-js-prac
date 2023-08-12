import { WebsitePhishingDataSet } from '@/data/WebsitePhishing';

export const websitePhishingTrain = async () => {
  const websitePhishingDataSet = new WebsitePhishingDataSet();
  await websitePhishingDataSet.loadData();
  websitePhishingDataSet.getTrainData();
};
