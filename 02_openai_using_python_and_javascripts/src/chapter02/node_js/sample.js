const readline = require('readline');
const { Configuration, OpenAIApi } = require('openai');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const api_key = "YOUR_OPENAI_API_KEY_HERE"; //☆
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

// function access_openai(prompt_value) {
//   const openai = new OpenAIApi(config);

//   openai.createCompletion({
//     model: "gpt-3.5-turbo-instruct",
//     prompt: prompt_value,
//     max_tokens: 100,
//   }).then(response=>{
//     const result = response.data.choices[0].text.trim();
//       console.log(result);
//   });
// }

function access_openai(prompt_value) {
  const openai = new OpenAIApi(config);

  openai.createCompletion({
    model: "gpt-3.5-turbo-instruct",
    prompt: "다음을 영어로 번역하십시오.\n\n" + prompt_value, //☆
    max_tokens: 100,
  }).then(response=>{
  const result = response.data.choices[0].text.trim();
    console.log(result);
  });
}