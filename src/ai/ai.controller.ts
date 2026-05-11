import { Controller, Get, Post, Put, Body } from '@nestjs/common';
import { AiService } from './ai.service';
import type { AiRequest } from './ai.service';

@Controller('ai')
export class AiController {
  constructor(private readonly aiService: AiService) {}

  @Post('generate')
  async generate(@Body() body: AiRequest) {
    const response = await this.aiService.generateResponse(body);
    return response;
  }

  @Post('analyze-screen')
  async analyzeScreen(
    @Body()
    body: {
      screenshotBase64: string;
      question?: string;
      model?: string;
    },
  ) {
    const response = await this.aiService.generateResponse({
      transcript:
        body.question ||
        'What is shown on this screen? Describe the code or content.',
      screenshotBase64: body.screenshotBase64,
      model: body.model || 'gemini-flash',
    });
    return response;
  }

  @Get('models')
  getModels() {
    return this.aiService.getAvailableModels();
  }

  @Put('settings')
  updateSettings(
    @Body()
    body: {
      geminiKey?: string;
      openaiKey?: string;
      anthropicKey?: string;
    },
  ) {
    const keys: Record<string, string> = {};
    if (body.geminiKey) keys.gemini = body.geminiKey;
    if (body.openaiKey) keys.openai = body.openaiKey;
    if (body.anthropicKey) keys.anthropic = body.anthropicKey;
    this.aiService.updateApiKeys(keys);
    return { message: 'Settings updated' };
  }
}
