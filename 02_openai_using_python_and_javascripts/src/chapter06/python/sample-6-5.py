import openai

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
    n=3,
    size="256x256"
  )
  for ob in response['data']:
    print("\n" + ob.url)


if __name__ == "__main__":
  input_text = input("텍스트를 입력: ")
  access_openai(input_text)
