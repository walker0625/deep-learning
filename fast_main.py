from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return '즐거운 웹'

@app.get("/image")
def param():
    return {'result' : '카리나'}

@app.get("/chatbot")
def param():
    return 'gpt'

@app.get("/video")
def param():
    return '넷플릭스'

# uvicorn fast_main:app --host 0.0.0.0 --port 8080 --reload
# http://172.16.20.132:8080/image