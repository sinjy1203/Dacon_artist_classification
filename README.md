# 월간 데이콘 예술 작품 화가 분류 AI 경진대회
>예술작품 이미지의 일부분인 데이터셋으로 올바르게 화가를 분류하는 대회

</br>

## 1. 제작 기간 & 참여 인원
- 2022년 10월 4일 ~ 
- 개인으로 참여

</br>

## 2. 사용 기술
- python
- pytorch
- CNN
- transfer learning
- augmentation

</br>

## 3. file 설명
`train.py` training model

`test.py` make submission file 

`model.py` custom model

`dataset.py` custom dataset

`utils.py` early stopping

`run_colab.ipynb`, 'run_colab_test.ipynb' 코랩 환경에서 gpu 활용

</br>

## 4. 트러블 슈팅
### 오버피팅 문제
- 학습데이터를 AlexNet 형태와 비슷한 모델의 가중치를 처음부터 학습하였더니 오버피팅 문제가 발생
- 분류할 클래스 수에 비해 학습이미지 데이터가 부족하다고 판단
- image augmentation으로 학습데이터수를 증가
- imagenet을 학습한 resnet, efficient net 파라미터 사용
- cutmix를 통한 학습 데이터 수 증가
### 과소적합 문제
- dropout, augmentation으로 인해 학습데이터의 loss가 줄어들지 않음 
- 학습 파라미터 수가 부족하다고 판단
- transfer learning에 freeze를 적용하지 않고 모든 파라미터를 학습
- resnet 뿐만 아니라 efficientnet, efficientnetv2 모델을 사용하여 학습
- imagenet에서 최근 좋은 성과를 내고 있는 다른 모델들을 찾아볼 예정
