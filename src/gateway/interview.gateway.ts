import {
  WebSocketGateway,
  WebSocketServer,
  SubscribeMessage,
  MessageBody,
  ConnectedSocket,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Logger } from '@nestjs/common';
import { Server, Socket } from 'socket.io';
import { AiService } from '../ai/ai.service';
import { TranscriptionService } from '../transcription/transcription.service';
import { RagService } from '../rag/rag.service';

@WebSocketGateway({
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
  transports: ['websocket', 'polling'],
})
export class InterviewGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  @WebSocketServer()
  server: Server;

  private readonly logger = new Logger(InterviewGateway.name);
  private clientSessions: Map<
    string,
    {
      transcription: ReturnType<
        TranscriptionService['createLiveTranscription']
      > | null;
      model: string;
      fullTranscript: string;
      lastScreenshot: string | null;
    }
  > = new Map();

  constructor(
    private aiService: AiService,
    private transcriptionService: TranscriptionService,
    private ragService: RagService,
  ) {}

  handleConnection(client: Socket) {
    this.logger.log(`Client connected: ${client.id}`);
    this.clientSessions.set(client.id, {
      transcription: null,
      model: 'nemotron',
      fullTranscript: '',
      lastScreenshot: null,
    });
  }

  handleDisconnect(client: Socket) {
    this.logger.log(`Client disconnected: ${client.id}`);
    const session = this.clientSessions.get(client.id);
    if (session?.transcription) {
      session.transcription.close();
    }
    this.clientSessions.delete(client.id);
  }

  @SubscribeMessage('audio-chunk')
  handleAudioChunk(
    @MessageBody() data: ArrayBuffer,
    @ConnectedSocket() client: Socket,
  ) {
    const session = this.clientSessions.get(client.id);
    if (!session) return;

    // Create transcription session on first audio chunk
    if (!session.transcription) {
      let accumulatedSentence = '';
      let silenceTimer: NodeJS.Timeout | null = null;
      let isGenerating = false;

      const triggerAi = async (questionText: string) => {
        const query = questionText.trim();
        if (!query || query.length < 3) return;
        if (isGenerating) {
          this.logger.log(`[Audio AI] Already generating, skipping duplicate query: "${query}"`);
          return;
        }
        isGenerating = true;
        this.logger.log(`[Audio AI] Triggering AI generation for: "${query}"`);
        client.emit('processing-start', { type: 'audio' });

        try {
          const response = await this.aiService.generateResponse({
            transcript: query,
            screenshotBase64: session.lastScreenshot || undefined,
            resumeContext: await this.ragService.getRelevantContext(query, session.lastScreenshot || undefined),
            model: session.model,
          });

          client.emit('ai-response', {
            text: response.text,
            done: true,
          });
        } catch (err: any) {
          this.logger.error(`AI generation failed: ${err.message || err}`);
          client.emit('error', { message: err.message || 'AI generation failed' });
        } finally {
          isGenerating = false;
        }
      };

      session.transcription = this.transcriptionService.createLiveTranscription(
        (text: string, isFinal: boolean, speechFinal?: boolean) => {
          // Send transcript update to client
          client.emit('transcript-update', { text, isFinal });

          if (isFinal) {
            accumulatedSentence = accumulatedSentence ? `${accumulatedSentence} ${text}` : text;
            session.fullTranscript += ' ' + text;

            // If speech_final is true (Deepgram detected end of speaker utterance), trigger AI immediately!
            if (speechFinal) {
              if (silenceTimer) clearTimeout(silenceTimer);
              const toSend = accumulatedSentence;
              accumulatedSentence = '';
              void triggerAi(toSend);
            } else {
              // Debounce trigger after 1.5s of silence
              if (silenceTimer) clearTimeout(silenceTimer);
              silenceTimer = setTimeout(() => {
                if (accumulatedSentence.trim()) {
                  const toSend = accumulatedSentence;
                  accumulatedSentence = '';
                  void triggerAi(toSend);
                }
              }, 1500);
            }
          }
        },
        (error: Error) => {
          client.emit('error', { message: error.message, type: 'transcription' });
        },
        () => {
          // On UtteranceEnd from Deepgram
          if (accumulatedSentence.trim()) {
            if (silenceTimer) clearTimeout(silenceTimer);
            const toSend = accumulatedSentence;
            accumulatedSentence = '';
            void triggerAi(toSend);
          }
        },
      );
    }

    // Forward audio to Deepgram
    session.transcription.sendAudio(Buffer.from(data));
  }

  @SubscribeMessage('audio-stop')
  handleAudioStop(@ConnectedSocket() client: Socket) {
    const session = this.clientSessions.get(client.id);
    if (session?.transcription) {
      session.transcription.close();
      session.transcription = null;
    }
  }

  @SubscribeMessage('screenshot')
  handleScreenshot(
    @MessageBody() data: { image: string },
    @ConnectedSocket() client: Socket,
  ) {
    const session = this.clientSessions.get(client.id);
    if (session) {
      session.lastScreenshot = data.image;
    }
  }

  @SubscribeMessage('analyze-screenshot')
  async handleAnalyzeScreenshot(
    @MessageBody() data: { image: string; question?: string },
    @ConnectedSocket() client: Socket,
  ) {
    const start = Date.now();
    const session = this.clientSessions.get(client.id);
    const model = session?.model || 'nemotron';

    if (session) {
      session.lastScreenshot = data.image;
    }

    client.emit('processing-start', { type: 'screenshot' });
    this.logger.log(`[analyze-screenshot] Starting analysis with model ${model}, image size: ${Math.round(data.image.length / 1024)} KB`);

    try {
      const apiStart = Date.now();
      const response = await this.aiService.generateResponse({
        transcript:
          data.question ||
          'What is shown on this screen? Describe the code or content.',
        screenshotBase64: data.image,
        resumeContext: await this.ragService.getRelevantContext(data.question || 'Analyze screen code and context', data.image),
        model,
      });
      this.logger.log(`[analyze-screenshot] generateResponse took ${Date.now() - apiStart}ms (latency reported: ${response.latency}ms)`);

      client.emit('ai-response', {
        text: response.text,
        done: true,
      });
      this.logger.log(`[analyze-screenshot] Completed full flow in ${Date.now() - start}ms`);
    } catch (error: any) {
      this.logger.error(`Failed to analyze screenshot after ${Date.now() - start}ms: ${error.message || error}`);
      client.emit('error', { message: error.message || 'Failed to analyze screenshot' });
      client.emit('ai-response', {
        text: '',
        done: true,
      });
    }
  }

  @SubscribeMessage('text-query')
  async handleTextQuery(
    @MessageBody() data: { text: string; model?: string },
    @ConnectedSocket() client: Socket,
  ) {
    const session = this.clientSessions.get(client.id);
    const model = data.model || session?.model || 'nemotron';

    client.emit('processing-start', {});

    const response = await this.aiService.generateResponse({
      transcript: data.text,
      screenshotBase64: session?.lastScreenshot || undefined,
      resumeContext: await this.ragService.getRelevantContext(data.text, session?.lastScreenshot || undefined),
      model,
    });

    client.emit('ai-response', {
      text: response.text,
      done: true,
    });
  }

  @SubscribeMessage('update-settings')
  handleUpdateSettings(
    @MessageBody()
    data: {
      model?: string;
      geminiKey?: string;
      openaiKey?: string;
      anthropicKey?: string;
      groqKey?: string;
      openrouterKey?: string;
      deepgramKey?: string;
      resumeContent?: string;
    },
    @ConnectedSocket() client: Socket,
  ) {
    const session = this.clientSessions.get(client.id);

    if (data.model && session) {
      session.model = data.model;
    }

    // Update API keys
    const apiKeys: Record<string, string> = {};
    if (data.geminiKey && data.geminiKey.trim()) {
      const cleanKey = data.geminiKey.trim();
      if (cleanKey.startsWith('sk-or-')) {
        this.logger.warn('Detected OpenRouter key passed in geminiKey field; routing to openrouterKey.');
        apiKeys.openrouter = cleanKey;
        this.ragService.updateOpenRouterKey(cleanKey);
      } else {
        apiKeys.gemini = cleanKey;
      }
    }
    if (data.openaiKey && data.openaiKey.trim()) apiKeys.openai = data.openaiKey.trim();
    if (data.anthropicKey && data.anthropicKey.trim()) apiKeys.anthropic = data.anthropicKey.trim();
    if (data.groqKey && data.groqKey.trim()) apiKeys.groq = data.groqKey.trim();
    if (data.openrouterKey && data.openrouterKey.trim()) {
      apiKeys.openrouter = data.openrouterKey.trim();
      this.ragService.updateOpenRouterKey(data.openrouterKey.trim());
    }

    if (Object.keys(apiKeys).length > 0) {
      this.aiService.updateApiKeys(apiKeys);
    }

    if (data.deepgramKey) {
      this.transcriptionService.updateApiKey(data.deepgramKey);
    }

    if (data.resumeContent) {
      this.ragService.updateResumeContext(data.resumeContent);
    }

    client.emit('settings-updated', { success: true });
    this.logger.log(`Settings updated for client ${client.id}`);
  }
}
