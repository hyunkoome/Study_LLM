const readline = require('readline');
const { Configuration, OpenAIApi } = require('openai');

const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆

const config = new Configuration({
  apiKey: api_key,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

(function(){
  rl.setPrompt("입력: ");
  rl.on("line", (line) => {
    if (line == "") {
      rl.close();
      return;
    }
    input_prompt(line);
  });
  rl.on("close", () => {
    console.log("close");
  });
  console.log("AI: 질문은 환영합니다.")
    rl.prompt(true);
})();

function input_prompt(msg) {
  const message = {
    "role": "user",
    "content": msg
  }
  messages.push(message);
  access_openai();
}

var messages = [
  {"role": "system", "content": "당신은 베이킹 어시스턴트입니다."}
]

async function access_openai() {
  const openai = new OpenAIApi(config);
  const response = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    messages: messages
  });
  const message = response.data.choices[0].message;
  messages.push(message);
  console.log("AI: " + message.content.trim());
  rl.prompt(true);
}
