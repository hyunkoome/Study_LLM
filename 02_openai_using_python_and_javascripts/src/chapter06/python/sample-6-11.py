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
  prompt = read_prompt("prompt.txt")

  response = openai.Image.create_edit(
    image=open("image.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt=prompt + prompt_value,
    n=1,
    size="256x256"
  )
  image_url = response['data'][0]['url']
  print(image_url)




if __name__ == "__main__":
  input_text = input("텍스트를 입력: ")
  access_openai(input_text)
