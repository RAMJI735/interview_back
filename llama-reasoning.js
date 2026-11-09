
require('dotenv').config();
const OpenAI = require('openai');

const apiKey = process.env.OPENROUTER_API_KEY;

if (!apiKey) {
  console.error('Error: OPENROUTER_API_KEY missing in .env');
  process.exit(1);
}

const openai = new OpenAI({
  apiKey,
  baseURL: 'https://openrouter.ai/api/v1',
  defaultHeaders: {
    'HTTP-Referer': 'https://github.com/RAMJI735/Interview_ai',
    'X-Title': 'Interview AI',
  },
});

// Free models only. No paid fallback.
const MODELS = [
  'cohere/north-mini-code:free',
  'nvidia/nemotron-3.5-lightning:free',
  'recursive/flint-v1.0:free',
];

async function runLlamaReasoningChat() {
  const question = "How many r's are in the word 'strawberry'?";

  let selectedModel = null;
  let response1 = null;

  // Try models one by one
  for (const model of MODELS) {
    try {
      console.log(`Trying free model: ${model}`);

      response1 = await openai.chat.completions.create({
        model,
        messages: [
          { role: 'user', content: question }
        ],
        reasoning: { enabled: true },
        max_tokens: 1500,
      });

      selectedModel = model;
      console.log(`\nConnected successfully: ${model}`);
      break;

    } catch (error) {
      console.error(`Model failed: ${model}`);
      console.error(error.status, error.message);

      // Try next model only for availability/rate/server errors
      if (![404, 429, 500, 502, 503, 504].includes(error.status)) {
        throw error;
      }
    }
  }

  if (!selectedModel) {
    throw new Error('All free models are unavailable. Try again later.');
  }

  const assistantMessage = response1.choices[0]?.message;

  console.log('\n[Turn 1]:');
  console.log(assistantMessage?.content);

  const messages = [
    { role: 'user', content: question },
    {
      role: 'assistant',
      content: assistantMessage?.content || '',
      ...(assistantMessage?.reasoning_details
        ? { reasoning_details: assistantMessage.reasoning_details }
        : {}),
    },
    {
      role: 'user',
      content: 'Are you sure? Think carefully and verify your answer.',
    },
  ];

  console.log('\n[Turn 2] Using:', selectedModel);

  const response2 = await openai.chat.completions.create({
    model: selectedModel,
    messages,
    reasoning: { enabled: true },
    max_tokens: 1500,
  });

  const finalResponse = response2.choices[0]?.message;

  console.log('\n[Final Answer]:');
  console.log(finalResponse?.content);

  return finalResponse;
}

if (require.main === module) {
  runLlamaReasoningChat().catch((error) => {
    console.error('\nChat failed:', error.message);
  });
}

module.exports = { runLlamaReasoningChat };
