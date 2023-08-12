import axios from 'axios';
import Papa from 'papaparse';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const parseCsv = async (data: any) => {
  return new Promise(resolve => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    data = data.map((row: any) => {
      return Object.keys(row).map(key => parseFloat(row[key]));
    });
    resolve(data);
  });
};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const shuffle = (data: any, target: any) => {
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

export const loadCsv = async (baseUrl: string, filename: string) => {
  try {
    const url = `${baseUrl}${filename}`;
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
