const readline = require('readline');
const { Configuration, OpenAIApi } = require('openai');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const api_key = "YOUR_OPENAI_API_KEY_HERE."; //☆

const config = new Configuration({
  apiKey: api_key,
});

input_prompt("텍스트를 입력하십시오: ");

function input_prompt(msg) {
  rl.question(msg, (inputText) => {
    rl.close();
    access_openai(inputText);
  });
}

const fs = require('fs')

function read_prompt(fname) {
  return fs.readFileSync(fname, 'utf-8');
}

const suffix_prompt = "\n\n1. ";

function access_openai(prompt_value) {
  const openai = new OpenAIApi(config);
  let prompt = read_prompt("prompt.txt");

  openai.createCompletion({
    model: "gpt-3.5-turbo-instruct",
    prompt: prompt + prompt_value,
    best_of: 5,
    max_tokens: 200,
  }).then(response=>{
    const result = response.data.choices[0].text.trim();
    console.log("\n결과: " + result );
  });
}


