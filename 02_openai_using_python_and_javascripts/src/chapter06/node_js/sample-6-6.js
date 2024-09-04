const fs = require("fs");
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

function read_prompt(fname) {
  return fs.readFileSync(fname, 'utf-8');
}

function input_prompt(msg) {
  rl.question(msg, (input_text) => {
    rl.close();
    access_openai(input_text);
  });
}

(function(){
  input_prompt("텍스트를 입력: ");
})();

function access_openai(prompt_value) {
  const openai = new OpenAIApi(config);
  const prompt = read_prompt("prompt.txt");
  openai.createImage({
    prompt: prompt + prompt_value,
    n: 3,
    size: "256x256",
  }).then(response=>{
    for (let ob of response.data.data) {
      console.log("\n" + ob.url);
    }
  });
}

