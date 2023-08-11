import { tf } from '@/constant/globalTf';

export const computeBaseline = ({
  trainTarget,
  testTarget,
  testName = '',
}: {
  trainTarget: tf.Tensor2D;
  testTarget: tf.Tensor2D;
  testName: string;
}) => {
  const avgPrice = trainTarget.mean().dataSync();
  console.log(`${testName} 평균 가격: `, avgPrice[0]);

  const baseline = testTarget.sub(avgPrice).square().mean().dataSync();
  const sqrtBaseLine = Math.sqrt(baseline[0]);
  console.log(`${testName} 기준 손실 From 평균 제곱오차: `, baseline[0]);
  console.log('기준 손실 제곱근: ', sqrtBaseLine);
  console.log(`목표 ${sqrtBaseLine} 보다 낮은 오차`);
};
