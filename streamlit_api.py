import datetime

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from streamlit_drawable_canvas import st_canvas

# streamlit run streamlit_api.py    

class_names = ["마동석","카리나","이수지"]

@st.cache_resource # 캐시에서 모델을 불러와서 사용
def load_model():
    
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512,3)
    model.load_state_dict(torch.load("./model_pth/mymodel.pth",map_location=torch.device('cpu')))
    model.eval()
    
    return model

#이미지 전처리
def transform_image(image):
    
    transforms_test = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ]
    )
    
    return transforms_test(image).unsqueeze(0) # 3,224,224

st.title('연예인 분류기 V.1')

upload_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

#upload_file = st.camera_input('웹캠')

if upload_file is not None:
    image = Image.open(upload_file).convert("RGB") # 흑백 사진일 경우 대비
    st.image(image, caption='업로드된 이미지', use_container_width=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    image.save(f'sent_data/{timestamp}.jpg') # 이렇게 요청 데이터를 따로 저장하는 것이 포트폴리오 어필 포인트!

    model = load_model()
    infer_image = transform_image(image)

    with torch.no_grad():
        result = model(infer_image)
        preds = torch.max(result,dim=1)[1]
        pred_classname = class_names[preds.item()]
        confidence = torch.softmax(result,dim=1)[0][preds.item()].item() * 100
    
    st.success(f'예측결과: {pred_classname} ({confidence:.2f}% 일치)')

# canvas_img = st_canvas(
    # fill_color='white',  # 내부색상 or RGB(+투명도)
    # stroke_width=3, # 펜 두께
    # stroke_color='black',
    # background_color='white',
    # height=400,
    # width=400,
    # drawing_mode='freedraw',
    # key='canvas' # session에 값을 저장
# )
# 
# if canvas_img is not None:
    # image = Image.fromarray(canvas_img.image_data).convert("RGB") # 흑백 사진일 경우 대비
    # st.image(image, caption='업로드된 이미지', use_container_width=True)
# 
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
    # image.save(f'sent_data/{timestamp}.jpg') # 이렇게 요청 데이터를 따로 저장하는 것이 포트폴리오 어필 포인트!
    # 
    # model = load_model()
    # infer_image = transform_image(image)
    # 
    # with torch.no_grad():
        # result = model(infer_image)
        # preds = torch.max(result,dim=1)[1]
        # pred_classname = class_names[preds.item()]
        # confidence = torch.softmax(result,dim=1)[0][preds.item()].item() * 100
    # 
    # st.success(f'예측결과: {pred_classname} ({confidence:.2f}% 일치)')

