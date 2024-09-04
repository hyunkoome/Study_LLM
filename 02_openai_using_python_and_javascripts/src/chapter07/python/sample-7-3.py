import openai
import base64 # 추가한다

api_key = "YOUR_OPENAI_API_KEY_HERE"

def read_prompt(fname):
  f = open(fname, encoding='utf-8')
  content = f.read()
  f.close()
  return content

def access_openai(prompt_value):
  openai.api_key = api_key
  response = openai.Completion.create(
    model="curie:ft-unclemos-2023-10-24-05-33-01",
    prompt=prompt_value,
    max_tokens=200)
  print(response.choices[0].text.strip())


if __name__ == "__main__":
  input_text = input("텍스트를 입력: ")
  access_openai(input_text)
