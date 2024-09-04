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

  response = openai.Image.create(
    prompt=prompt + prompt_value,
    n=1,
    size="256x256",
    response_format="b64_json"
  )
  image_b64 = response['data'][0]["b64_json"]
  binary_data = base64.b64decode(image_b64)
  with open("created_image.png", "wb") as f:
    f.write(binary_data)
  print("파일에 저장했습니다.")



if __name__ == "__main__":
  input_text = input("텍스트를 입력: ")
  access_openai(input_text)
