import { map, meanBy } from 'lodash';
import { Vector } from './type';

export const normalizeVector = (vector: Vector, vectorMean: number, vectorStddev: number) => {
  return map(vector, (x) => (x - vectorMean) / vectorStddev);
};

export const stddev = (vector: Vector) => {
  let squareSum = 0;
  const vectorMean = meanBy(vector);
  for (const x of vector) {
    squareSum += (x - vectorMean) * (x - vectorMean);
  }
  return Math.sqrt(squareSum / (vector.length - 1));
};
