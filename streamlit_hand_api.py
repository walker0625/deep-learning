import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- 1. 모델 로딩 및 설정 ---
@st.cache_resource
def load_model():
    
    model = nn.Sequential(
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    
    # try:
        #CPU 환경에서 모델 가중치 로드
        # model.load_state_dict(torch.load('model_pth/model_hand.pth', map_location=torch.device('cpu')))
    # except FileNotFoundError:
        # st.error(f"'{model_path}' 파일을 찾을 수 없습니다. 모델 가중치 파일을 확인해주세요.")
        # return None
        
    model.eval() # 추론 모드로 설정
    
    return model

# --- 2. 이미지 전처리 설정 ---

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 흑백으로 변환
    transforms.Resize((28, 28)),             # MNIST 모델 입력 사이즈에 맞게 조정
    transforms.ToTensor(),                         # PyTorch 텐서로 변환
])


# --- 3. Streamlit UI 구성 ---

st.title("손글씨 분류")
st.write("0-9 숫자를 손으로 그리시오")

# 캔버스 UI 설정
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)", 
    stroke_width=15,                     
    stroke_color="#FFFFFF",             
    background_color="#000000",         
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 모델 로드
model = load_model()

# "예측하기" 버튼
if st.button("예측하기"):
    if canvas_result.image_data is not None and model is not None:
        # 캔버스에서 그린 이미지를 PIL Image로 변환 (알파 채널 제외)
        img_pil = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('RGB')
        
        # 이미지 전처리
        img_transformed = transform(img_pil)
        
        # --- 4. 예측 수행 ---
        with torch.no_grad():
            # 모델 입력에 맞게 1차원 벡터로 펼치기
            image_flattened = img_transformed.view(-1, 28 * 28)
            outputs = model(image_flattened)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        st.success(f"🤖 예측된 숫자: {prediction}")
        
        # 예측 결과와 함께 원본 이미지 출력
        st.image(img_pil, caption='입력한 이미지', use_column_width=True)
    elif model is None:
        st.warning("모델이 로드되지 않아 예측을 수행할 수 없습니다.")
    else:
        st.warning("숫자를 먼저 그려주세요.")