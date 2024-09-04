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
  openai.createImage({
    prompt: prompt_value,
    n: 1,
    size: "256x256",
  }).then(response=>{
    const image_url = response.data.data[0].url;
    console.log(image_url);
  }).catch(reason=>{
    const err = new String(reason);
    console.log("에러가 발생했습니다: " + err);
  });
}






