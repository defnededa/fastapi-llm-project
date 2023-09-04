from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI , Request 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

templates = Jinja2Templates(directory= "html")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#check
@app.get("/")
def read_root():
    return {"Hello": "World"}

#HTML
@app.get("/index/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#path to use
model_path3 = "./gpt2_model/"
tokenizer3 = AutoTokenizer.from_pretrained(model_path3)
model3 = AutoModelForCausalLM.from_pretrained(model_path3)

#prompt class
class Prompt(BaseModel):
    string: str

#enter prompt
@app.post("/promptgpt2")
def enter_prompt_gpt2(prompt: Prompt ):
    prompt = prompt.string
    input_ids = tokenizer3.encode(prompt, return_tensors='pt')
    output = model3.generate(inputs=input_ids,
                            max_length=100,
                            do_sample=True,
                            top_k=30,
                            pad_token_id= tokenizer3.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer3.decode(output[0], skip_special_tokens=True)





