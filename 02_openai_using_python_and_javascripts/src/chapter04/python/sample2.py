import openai

api_key = "YOUR_OPENAI_API_KEY_HERE"

openai.api_key = api_key
for ob in openai.Model.list().data:
  print(ob.id)

