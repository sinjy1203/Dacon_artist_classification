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
`run_colab` 코랩 환경에서 gpu로 학습

</br>

## 4. 트러블 슈팅
### 오버피팅 문제
- 분류할 클래스 수에 비해 학습이미지 데이터가 부족하여 오버피팅 문제가 발생
- image augmentation으로 학습데이터수를 늘려 어느정도 해결
- resnet, efficient net을 활용한 transfer learning을 사용
### 정확도 부족
- cutmix augmentation과 더 큰 efficient net 모델을 활용할 예정
