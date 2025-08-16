# 딥러닝

## Process(Image 분류)

### 1. Model 생성

- #### 1) Data(Image) 준비 및 변환
  - data_transform 세팅 : w*h size, 정규화, 반전

  - channel 순서 변경 

- #### 2) DataLoader로 변환
  - train/test 각각 따로

  - batch_size, shuffle 설정

- #### 3) Model의 Layer 값을 바꿔가며 실험
  - 임의로 값을 바꿔가며, 결과값으로 성능 확인하며 반복

  - 맨처음 Layer 입력값(in_channels)과 분류기의 첫 입력값(in_feature - Layer out_channels * h * w), 마지막 출력값(out_features 값 - class 수)은 임의값이 아니라 공식과 필요에 맞게 지정해야 함

- #### 4) Model 학습  
  - 필요값 지정 : Learning Rate(0.001~0.003), epoch 수, 최적화 함수(Adam), 손실 함수(분류:CrossEntropyLoss), gpu 사용 여부, tensorboard

  - 학습 실행 : 가중치, 손실, 순전파, 역전파

### 2. 전이 학습(fine tuning)

  - #### 1) Data(Image) 준비 및 변환

- #### 2) DataLoader로 변환

- #### 3) Model의 기존 학습 가중치 활용 및 Layer Frozen 결정
  - pretrained=True(image-net 등) : 기존 가중치 활용 
  - param.requires_grad = False : Frozen(가중치 기존 그대로 사용)
  - 분류기 변경의 경우 첫 in_features와 마지막 out_features(class 수) 수정

- #### 4) Model 학습  

  - Frozen된 Layer는 학습되지 않고, 새롭게 설정한 Layer(분류기 등)만 학습을 통해 업데이트 됨