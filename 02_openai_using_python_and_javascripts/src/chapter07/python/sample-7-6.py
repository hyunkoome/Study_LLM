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
  try:
    response = openai.Image.create(
      prompt=prompt_value,
      n=1,
      size="256x256"
    )
    image_url = response['data'][0]['url']
    print(image_url)
  except openai.error.InvalidRequestError as e:
    print(f"유효하지 않은 요청이 전송되었습니다: {e}")
    pass
  except:
    print(f"에러가 발생했습니다.")
    pass

if __name__ == "__main__":
  input_text = input("텍스트를 입력: ")
  access_openai(input_text)
