import OpenAI from 'openai';
import { Logger } from '@nestjs/common';

export const REASONING_MODELS = [
  'cohere/north-mini-code:free',
  'nvidia/nemotron-3.5-lightning:free',
  'recursive/flint-v1.0:free',
  'meta-llama/llama-3.3-70b-instruct:free',
  'meta-llama/llama-3.1-8b-instruct:free',
];

export interface LlamaReasoningOptions {
  apiKey: string;
  systemPrompt: string;
  userPrompt: string;
  screenshotBase64?: string;
  enableSecondTurnVerification?: boolean;
}

export async function runLlamaReasoning(
  options: LlamaReasoningOptions,
  logger?: Logger,
): Promise<string> {
  const {
    apiKey,
    systemPrompt,
    userPrompt,
    screenshotBase64,
    enableSecondTurnVerification = true,
  } = options;

  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY is not configured for Llama Reasoning.');
  }

  const openai = new OpenAI({
    apiKey,
    baseURL: 'https://openrouter.ai/api/v1',
    defaultHeaders: {
      'HTTP-Referer': 'https://github.com/RAMJI735/Interview_ai',
      'X-Title': 'Interview AI',
    },
  });

  const baseMessages: any[] = [];
  if (systemPrompt) {
    baseMessages.push({ role: 'system', content: systemPrompt });
  }

  const userContent: any = screenshotBase64
    ? [
        {
          type: 'text',
          text: userPrompt + '\n\nHere is a screenshot of the screen for context:',
        },
        {
          type: 'image_url',
          image_url: { url: screenshotBase64, detail: 'low' },
        },
      ]
    : userPrompt;

  baseMessages.push({ role: 'user', content: userContent });

  let selectedModel: string | null = null;
  let response1: any = null;
  let lastError: any = null;

  for (const model of REASONING_MODELS) {
    try {
      logger?.log(`[Llama Reasoning] Trying free model: ${model}`);

      response1 = await openai.chat.completions.create({
        model,
        messages: baseMessages,
        max_tokens: 1500,
        ...({ reasoning: { enabled: true } } as any),
      });

      selectedModel = model;
      logger?.log(`[Llama Reasoning] Connected successfully: ${model}`);
      break;
    } catch (error: any) {
      logger?.warn(
        `[Llama Reasoning] Model failed: ${model} - ${error.status || error.message}`,
      );
      lastError = error;

      // If vision input failed (e.g. 400 Bad Request: image not supported), retry text-only
      if (
        screenshotBase64 &&
        (error.status === 400 || error.message?.toLowerCase().includes('image'))
      ) {
        try {
          const textOnlyMessages = [
            ...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []),
            { role: 'user', content: userPrompt },
          ];
          response1 = await openai.chat.completions.create({
            model,
            messages: textOnlyMessages,
            max_tokens: 1500,
            ...({ reasoning: { enabled: true } } as any),
          });
          selectedModel = model;
          logger?.log(
            `[Llama Reasoning] Connected successfully (text fallback): ${model}`,
          );
          break;
        } catch (textErr: any) {
          lastError = textErr;
        }
      }

      // Try next model for availability/rate/server errors
      if (
        error.status &&
        ![400, 404, 429, 500, 502, 503, 504].includes(error.status)
      ) {
        throw error;
      }
    }
  }

  if (!selectedModel || !response1) {
    throw (
      lastError ||
      new Error('All free reasoning models are unavailable. Try again later.')
    );
  }

  const assistantMessage = response1.choices[0]?.message;
  const turn1Content = assistantMessage?.content || '';

  if (!enableSecondTurnVerification) {
    return turn1Content;
  }

  // Turn 2: Reasoning verification turn as shown in llama-reasoning.js
  try {
    const turn2Messages: any[] = [
      ...baseMessages,
      {
        role: 'assistant',
        content: turn1Content,
        ...(assistantMessage?.reasoning_details
          ? { reasoning_details: assistantMessage.reasoning_details }
          : {}),
      },
      {
        role: 'user',
        content: 'Are you sure? Think carefully and verify your answer.',
      },
    ];

    logger?.log(
      `[Llama Reasoning Turn 2] Verifying answer using: ${selectedModel}`,
    );
    const response2 = await openai.chat.completions.create({
      model: selectedModel,
      messages: turn2Messages,
      max_tokens: 1500,
      ...({ reasoning: { enabled: true } } as any),
    });

    const finalAnswer = response2.choices[0]?.message?.content;
    if (finalAnswer && finalAnswer.trim().length > 0) {
      return finalAnswer;
    }
  } catch (turn2Err: any) {
    logger?.warn(
      `[Llama Reasoning Turn 2] Verification failed: ${turn2Err.message}. Using Turn 1 answer.`,
    );
  }

  return turn1Content;
}

