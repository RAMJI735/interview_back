import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { GoogleGenerativeAI, Part } from '@google/generative-ai';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { TextBlock } from '@anthropic-ai/sdk/resources/messages';
import { runLlamaReasoning } from './llama-reasoning';

export interface AiRequest {
  transcript: string;
  screenshotBase64?: string;
  resumeContext?: string;
  model: string;
}

export interface AiResponse {
  text: string;
  model: string;
  latency: number;
}

@Injectable()
export class AiService {
  private readonly logger = new Logger(AiService.name);

  // API keys can be updated at runtime from client
  private apiKeys: Record<string, string> = {};

  constructor(private configService: ConfigService) {
    this.apiKeys = {
      gemini: this.configService.get<string>('GEMINI_API_KEY') || '',
      openai: this.configService.get<string>('OPENAI_API_KEY') || '',
      anthropic: this.configService.get<string>('ANTHROPIC_API_KEY') || '',
      openrouter: this.configService.get<string>('OPENROUTER_API_KEY') || '',
      groq: this.configService.get<string>('GROQ_API_KEY') || '',
    };
  }

  updateApiKeys(keys: Record<string, string>) {
    this.apiKeys = { ...this.apiKeys, ...keys };
    this.logger.log('API keys updated');
  }

  async generateResponse(request: AiRequest): Promise<AiResponse> {
    const startTime = Date.now();
    let text: string;

    const systemPrompt = this.buildSystemPrompt(request.resumeContext);
    const userPrompt = this.buildUserPrompt(
      request.transcript,
      !!request.screenshotBase64,
    );

    try {
      switch (request.model) {
        case 'gemini-flash':
          text = await this.callGemini(
            systemPrompt,
            userPrompt,
            request.screenshotBase64,
          );
          break;
        case 'gpt-4o':
          text = await this.callOpenAI(
            systemPrompt,
            userPrompt,
            request.screenshotBase64,
          );
          break;
        case 'claude-sonnet':
          text = await this.callClaude(systemPrompt, userPrompt);
          break;
        case 'nemotron':
        case 'openrouter-nemotron':
          text = await this.callOpenRouter(
            systemPrompt,
            userPrompt,
            request.screenshotBase64,
          );
          break;
        case 'llama-reasoning':
        case 'groq-llama':
        case 'llama':
          text = await this.callLlamaReasoning(
            systemPrompt,
            userPrompt,
            request.screenshotBase64,
          );
          break;
        default:
          if (this.apiKeys.openrouter) {
            this.logger.log(`Model "${request.model}" routing to OpenRouter Nemotron`);
            text = await this.callOpenRouter(
              systemPrompt,
              userPrompt,
              request.screenshotBase64,
            );
          } else {
            text = await this.callGemini(
              systemPrompt,
              userPrompt,
              request.screenshotBase64,
            );
          }
      }
    } catch (error: unknown) {
      const errMsg = error instanceof Error ? error.message : 'Unknown error';
      this.logger.error(`AI generation failed: ${errMsg}`);
      text = `Error: ${errMsg}. Please check your API key for ${request.model}.`;
    }

    return {
      text,
      model: request.model,
      latency: Date.now() - startTime,
    };
  }

  // private buildSystemPrompt(resumeContext?: string): string {
  //   let prompt =
  //     'You are an expert AI interview assistant helping a Full Stack MERN developer during a live job interview.\n' +
  //     'Your responses should be:\n' +
  //     '- Concise but comprehensive\n' +
  //     '- Well-structured with bullet points and code examples when relevant\n' +
  //     '- Natural-sounding (as if the candidate is speaking)\n' +
  //     '- Technically accurate and up-to-date\n' +
  //     '- Formatted in Markdown for readability\n\n' +
  //     'IMPORTANT: Keep answers brief enough to read quickly during an interview (aim for 30-60 seconds of reading time).\n' +
  //     'If the question is about coding, provide clean, working code with brief explanations.';

  //   if (resumeContext) {
  //     prompt +=
  //       `\n\nHere is the candidate's resume/background for personalization:\n` +
  //       resumeContext;
  //   }

  //   return prompt;
  // }

  private buildSystemPrompt(resumeContext?: string): string {
    let prompt =
      'You are an expert MCA university exam assistant.\n' +
      'CORE RULES:\n' +
      '1. DIRECT ANSWERS ONLY: Directly solve and write the exam answer for every question shown/asked. Never summarize, describe the image, or repeat questions without answers.\n' +
      '2. EXAM STRUCTURE: Definition -> Key Points / Headings -> Working Code / Example / Diagram -> Brief Conclusion.\n' +
      '3. CONCISE & HIGH-SCORING: No conversational filler or chit-chat. Provide technically accurate, exam-ready answers that can be written directly on an MCA answer sheet.\n' +
      '4. MULTI-QUESTION: If multiple questions appear, answer each in order under its question label (e.g. Q1, Q2(a)).';

    if (resumeContext) {
      prompt += `\n\nContext:\n${resumeContext}`;
    }

    return prompt;
  }

  private buildUserPrompt(
    transcript: string,
    hasScreenshot: boolean = false,
  ): string {
    const text = transcript?.trim();

    // Screenshot is available: prioritize solving visible content
    if (hasScreenshot) {
      return `
IMPORTANT: An image/screenshot is attached.

Your task is to READ and SOLVE the actual questions visible in the screenshot.

STRICT INSTRUCTIONS:
1. Never describe the image or screen.
2. Never explain what objects or content are visible.
3. Extract all readable questions from the screenshot.
4. Write complete answers to every visible question.
5. Answer questions in their original order.
6. Use proper headings, definitions, examples and explanations.
7. For numerical questions, show step-by-step calculations.
8. For accounting questions, include formats and examples.
9. Write in simple, exam-ready English.
10. Do not skip any question.

The transcript below may contain an automatically generated description of the screenshot. Treat it ONLY as supporting context, NOT as the actual question.

Supporting context:
${text || 'No transcript available.'}

NOW SOLVE ALL QUESTIONS DIRECTLY.
`;
    }

    // No screenshot: answer the actual spoken/typed question
    return `
Answer the following question directly.

Do not describe the screen.
Provide a complete, accurate and exam-ready answer.

Question:
${text}

Use headings, examples and explanations wherever needed.
`;
  }

  // ── OpenRouter (NVIDIA Nemotron with Vision & Fallbacks) ──
  private async callOpenRouter(
    systemPrompt: string,
    userPrompt: string,
    screenshotBase64?: string,
    preferredModel?: string,
  ): Promise<string> {
    const apiKey = this.apiKeys.openrouter;
    if (!apiKey) {
      throw new Error(
        'OpenRouter API key is not configured. Please enter your OpenRouter key in Settings.',
      );
    }

    const openai = new OpenAI({
      apiKey,
      baseURL: 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': 'https://github.com/RAMJI735/Interview_ai',
        'X-Title': 'Interview AI',
      },
    });

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: 'system', content: systemPrompt },
    ];

    if (screenshotBase64) {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'text',
            text: userPrompt + '\n\nHere is a screenshot of the screen for context:',
          },
          {
            type: 'image_url',
            image_url: { url: screenshotBase64, detail: 'low' },
          },
        ],
      });
    } else {
      messages.push({ role: 'user', content: userPrompt });
    }

    // Try Nemotron models on OpenRouter (free tier priority)
    const modelCandidates = [
      ...(preferredModel ? [preferredModel] : []),
      'google/gemini-2.0-flash-exp:free',
      'nvidia/llama-3.1-nemotron-70b-instruct:free',
      'nvidia/nemotron-3.5-super-vl:free',
      'nvidia/nemotron-3-ultra-550b-a55b:free',
      'meta-llama/llama-3.3-70b-instruct:free',
    ];

    let lastErr: Error | null = null;
    for (const model of modelCandidates) {
      try {
        this.logger.log(`Attempting OpenRouter generation with model: ${model}`);
        const completion = await openai.chat.completions.create({
          model,
          messages,
          max_tokens: 4096,
          temperature: 0.7,
        });

        const content = completion.choices[0]?.message?.content;
        if (content) {
          return content;
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        this.logger.warn(`OpenRouter model ${model} failed: ${msg}`);
        lastErr = err instanceof Error ? err : new Error(msg);
      }
    }

    throw lastErr || new Error('All OpenRouter model candidates failed');
  }

  // ── Llama Reasoning (OpenRouter Free Reasoning with Verification & Groq/Gemini Fallbacks) ──
  private async callLlamaReasoning(
    systemPrompt: string,
    userPrompt: string,
    screenshotBase64?: string,
  ): Promise<string> {
    if (this.apiKeys.openrouter) {
      try {
        return await runLlamaReasoning(
          {
            apiKey: this.apiKeys.openrouter,
            systemPrompt,
            userPrompt,
            screenshotBase64,
          },
          this.logger,
        );
      } catch (err: any) {
        this.logger.warn(
          `Llama Reasoning via OpenRouter failed: ${err?.message || err}. Falling back to Groq/Gemini...`,
        );
      }
    }

    // For screenshots, Groq is text-only, so always use Gemini Vision
    if (!screenshotBase64 && this.apiKeys.groq) {
      try {
        return await this.callGroq(systemPrompt, userPrompt);
      } catch (groqErr: any) {
        this.logger.warn(`Groq fallback failed: ${groqErr?.message || groqErr}`);
      }
    }

    return this.callGemini(systemPrompt, userPrompt, screenshotBase64);
  }

  // ── Groq (Llama Fast with Model Fallbacks) ──
  private async callGroq(
    systemPrompt: string,
    userPrompt: string,
  ): Promise<string> {
    const apiKey = this.apiKeys.groq;
    if (!apiKey) {
      this.logger.warn('Groq API key not configured, falling back to Gemini Flash');
      return this.callGemini(systemPrompt, userPrompt);
    }

    const groqClient = new OpenAI({
      apiKey,
      baseURL: 'https://api.groq.com/openai/v1',
    });

    const candidateModels = [
      'llama-3.3-70b-versatile',
      'llama-3.1-8b-instant',
      'llama3-70b-8192',
      'llama3-8b-8192',
    ];

    let lastError: any = null;
    for (const model of candidateModels) {
      try {
        this.logger.log(`Attempting Groq completion with model: ${model}`);
        const completion = await groqClient.chat.completions.create({
          model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          max_tokens: 1024,
          temperature: 0.7,
        });

        const content = completion.choices[0]?.message?.content;
        if (content) {
          return content;
        }
      } catch (err: any) {
        this.logger.warn(`Groq model ${model} failed: ${err.message}`);
        lastError = err;
      }
    }

    // If OpenRouter is available, fallback to OpenRouter
    if (this.apiKeys.openrouter) {
      this.logger.log('All Groq models failed, falling back to OpenRouter...');
      return this.callOpenRouter(systemPrompt, userPrompt);
    }

    throw lastError || new Error('All Groq models failed');
  }

  // ── Gemini (with Vision support & OpenRouter fallback) ──
  private async callGemini(
    systemPrompt: string,
    userPrompt: string,
    screenshotBase64?: string,
  ): Promise<string> {
    const envKey = (this.configService.get<string>('GEMINI_API_KEY') || process.env.GEMINI_API_KEY || '').trim();
    let apiKey = (this.apiKeys.gemini || '').trim();

    // If apiKey is empty or mistakenly set to OpenRouter key, use the working envKey
    if (!apiKey || apiKey.startsWith('sk-or-')) {
      apiKey = envKey;
    }

    // Function to run Gemini with a given key
    const runGoogleApi = async (keyToUse: string): Promise<string> => {
      const genAI = new GoogleGenerativeAI(keyToUse);
      const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
      const parts: Part[] = [{ text: `${systemPrompt}\n\n${userPrompt}` }];

      if (screenshotBase64) {
        const mimeMatch = screenshotBase64.match(/^data:(image\/\w+);base64,/);
        const mimeType = mimeMatch ? mimeMatch[1] : 'image/png';
        const base64Data = screenshotBase64.replace(/^data:image\/\w+;base64,/, '');
        parts.push({
          inlineData: {
            mimeType: mimeType,
            data: base64Data,
          },
        });
        const firstPart = parts[0] as { text: string };
        firstPart.text += '\n\nHere is a screenshot of the screen for additional context:';
      }

      const result = await model.generateContent({
        contents: [{ role: 'user', parts }],
        generationConfig: {
          // @ts-ignore
          thinkingConfig: { thinkingBudget: 0 },
        },
      });
      return result.response.text();
    };

    try {
      this.logger.log(`Attempting Google Gemini with key (prefix: ${apiKey.substring(0, 8)}..., len: ${apiKey.length})`);
      return await runGoogleApi(apiKey);
    } catch (err: any) {
      this.logger.warn(`Primary Gemini call failed: ${err.message}`);

      // If primary key failed and envKey is different, try envKey
      if (envKey && envKey !== apiKey) {
        try {
          this.logger.log(`Retrying Google Gemini with .env key (prefix: ${envKey.substring(0, 8)}...)`);
          return await runGoogleApi(envKey);
        } catch (retryErr: any) {
          this.logger.warn(`Retry with .env key also failed: ${retryErr.message}`);
        }
      }

      // If OpenRouter key is configured, fallback to OpenRouter Gemini
      if (this.apiKeys.openrouter) {
        this.logger.log('Falling back to OpenRouter Google Gemini 2.0 Flash...');
        return this.callOpenRouter(systemPrompt, userPrompt, screenshotBase64, 'google/gemini-2.0-flash-exp:free');
      }

      throw err;
    }
  }

  // ── OpenAI GPT-4o ──
  private async callOpenAI(
    systemPrompt: string,
    userPrompt: string,
    screenshotBase64?: string,
  ): Promise<string> {
    const openai = new OpenAI({ apiKey: this.apiKeys.openai });

    const messages: OpenAI.ChatCompletionMessageParam[] = [
      { role: 'system', content: systemPrompt },
    ];

    if (screenshotBase64) {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'text',
            text: userPrompt + '\n\nHere is a screenshot for context:',
          },
          {
            type: 'image_url',
            image_url: { url: screenshotBase64, detail: 'low' },
          },
        ],
      });
    } else {
      messages.push({ role: 'user', content: userPrompt });
    }

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages,
      max_tokens: 1024,
      temperature: 0.7,
    });

    return completion.choices[0]?.message?.content || 'No response generated.';
  }

  // ── Claude Sonnet ──
  private async callClaude(
    systemPrompt: string,
    userPrompt: string,
  ): Promise<string> {
    const anthropic = new Anthropic({
      apiKey: this.apiKeys.anthropic,
    });

    const message = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-20241022',
      max_tokens: 1024,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
    });

    const textBlock = message.content.find(
      (block): block is TextBlock => block.type === 'text',
    );
    return textBlock ? textBlock.text : 'No response generated.';
  }

  getAvailableModels() {
    return [
      {
        id: 'nemotron',
        name: 'NVIDIA Nemotron 3.5',
        description: 'Free + OpenRouter',
        available: !!this.apiKeys.openrouter,
      },
      {
        id: 'llama-reasoning',
        name: 'Llama Reasoning',
        description: 'Free + Reasoning',
        available: !!this.apiKeys.openrouter || !!this.apiKeys.groq,
      },
      {
        id: 'gemini-flash',
        name: 'Gemini 2.5 Flash',
        description: 'Fast + Vision',
        available: !!this.apiKeys.gemini,
      },
      {
        id: 'gpt-4o',
        name: 'GPT-4o',
        description: 'Complex Reasoning',
        available: !!this.apiKeys.openai,
      },
      {
        id: 'claude-sonnet',
        name: 'Claude 3.5 Sonnet',
        description: 'Clean Code',
        available: !!this.apiKeys.anthropic,
      },
    ];
  }
}
