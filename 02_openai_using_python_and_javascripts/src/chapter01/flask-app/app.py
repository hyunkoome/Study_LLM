from flask import Flask, render_template, request
import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"  # ☆

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html', question=None, result=None)

@app.route('/', methods=['POST'])
def submit():
  prompt = request.form['prompt']
  result = access_openai(prompt)
  return render_template('index.html', question=prompt, result=result)

def access_openai(prompt_value):
  openai.api_key = api_key
  response = openai.Completion.create(model="gpt-3.5-turbo-instruct",
    prompt=prompt_value, max_tokens=100,
    n=2, stop=None, temperature=0.5)
  return response.choices[0].text.strip()
