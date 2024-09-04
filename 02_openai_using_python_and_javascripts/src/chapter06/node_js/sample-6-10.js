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
    n: 1,
    size: "256x256",
    response_format:"b64_json"
  }).then(response=>{
    const image_b64 = response.data.data[0]['b64_json'];
    const binary_data = Buffer.from(image_b64, 'base64');
    fs.writeFileSync('created_image.png', binary_data);
    console.log("파일에 저장했습니다. ");
  });
}


