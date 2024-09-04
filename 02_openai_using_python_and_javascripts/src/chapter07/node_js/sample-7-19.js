const fs = require("fs");
const readline = require('readline');
const { Configuration, OpenAIApi } = require('openai');
const similarity = require('compute-cosine-similarity');

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
  const jsonf = fs.readFileSync('sample_data.json', 'utf-8');
  const json_data = JSON.parse(jsonf);

  const openai = new OpenAIApi(config);
  
  openai.createEmbedding({
    model:"text-embedding-ada-002",
    input:prompt_value
  }).then(response=>{
    const emb_data = response.data.data[0].embedding;
    let result = [];
    for(let ob of json_data) {
      res = {
        'title': ob['title'],
        'sim': similarity(ob['embedding'], emb_data)
      };
      result.push(res);
    }
    result = result.sort((a,b)=> a.sim > b.sim);
    for(let ob of result) {
      console.log(ob['title'] + ' (' + ob['sim'] + ')');
    }
  });
}








