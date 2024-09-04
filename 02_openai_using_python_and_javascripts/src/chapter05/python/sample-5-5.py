import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"

messages=[
  {"role": "system", "content": "당신은 베이킹 어시스턴트입니다."},
  {"role": "user", "content": "안녕."},
  {"role": "assistant", "content": "하이! 저는 베이킹 전문 어시스턴트예요♥ 무엇이든 대답해드려요♥"},
  {"role": "user", "content": "베이킹으로 무언가 만들고 싶어."},
  {"role": "assistant", "content": "와! 베이킹을 하고 싶다니 멋진데♥ 무엇을 만들고 싶은지 알려 줄래?"}
]

def access_openai():
  openai.api_key = api_key
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
  )
  message = response.choices[0].message
  messages.append(message)
  print("AI: " + message.content.strip())

if __name__ == "__main__":
  print("AI: 질문을 환영합니다.")
  while True:
    input_text = input("입력: ")
    if input_text == "":
      print("close")
      break
    message = {
      "role": "user",
      "content": input_text
    }
    messages.append(message)
    access_openai()


