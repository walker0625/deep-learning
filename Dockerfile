# 베이스 이미지 선택
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install uvicorn fastapi

COPY infer_main.py .
COPY docker_model/ /app/model/
COPY docker_img/ /app/model/

EXPOSE 7070
CMD ["uvicorn", "infer_main:app", "--host", "0.0.0.0", "--port", "7070"]