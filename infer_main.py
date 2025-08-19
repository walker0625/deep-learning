from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image

import json
import datetime

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# <전체 흐름>

# 1. Model 로드

# 2. 클라이언트에게 받은 이미지 transforms(tensor/numpy)으로 변환 - model에 맞게 전처리

# *** mlops/aiops의 경우 사용자 요청 이미지를 따로 저장(S3/DISK)하여 학습용으로 활용 
# *** 1주일/1달 단위 활용하는 것으로 포트폴리오에서 설계 어필!

# 3. 전처리 결과물 Model로 추론

# 4. 추론 결과를 클라이언트에게 전달

app = FastAPI(title='ResNet34 Inference API')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.resnet34(pretrained=True)
model.fc = nn.Linear(in_features=512, out_features=3, bias=True) # 학습 할 때 파라미터로 통일
model.load_state_dict(torch.load('model_pth/mymodel.pth')) # 가중치 세팅
model.eval()
model.to(device)

# 학습할 때와 같은 값 변환
transforms_infer = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]
)

class predict_response(BaseModel):
    name : str
    score : float
    type : int
    
@app.post("/predict", response_model=predict_response)    
async def predict(image_file: UploadFile=File(...)):  
    image = Image.open(image_file.file)
    
    # uuid, index, timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    image.save(f'sent_data/{timestamp}.jpg') # 이렇게 요청 데이터를 따로 관리하는 것이 포트폴리오 어필 포인트!
    img_tensor = transformed_image = transforms_infer(image).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    with torch.no_grad():
        pred = model(img_tensor)
    
    pred_result = torch.max(pred, dim=1)[1].item() # 0, 1, 2
    pred_score = nn.Softmax()(pred)[0] # 전체를 0 ~ 1로 봤을 때 결과값을 그 사이 값으로 변환해줌 
    classname = ['마동석', '카리나', '이수지']
    
    return predict_response(name=classname[pred_result], score=float(pred_score[pred_result]), type=pred_result) 

@app.post("/dummy-predict", response_model=predict_response)    
async def dummy_predict(image_file: UploadFile=File(...)): 
    return predict_response(name='test', score=0.96, type=1) 