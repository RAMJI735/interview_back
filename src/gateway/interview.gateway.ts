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
      model: 'gemini-flash',
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
      session.transcription = this.transcriptionService.createLiveTranscription(
        (text: string, isFinal: boolean) => {
          // Send transcript update to client
          client.emit('transcript-update', { text, isFinal });

          if (isFinal) {
            session.fullTranscript += ' ' + text;

            // Trigger AI response on final transcript
            client.emit('processing-start', {});

            void this.aiService
              .generateResponse({
                transcript: text,
                screenshotBase64: session.lastScreenshot || undefined,
                resumeContext: this.ragService.getContext(),
                model: session.model,
              })
              .then((response) => {
                client.emit('ai-response', {
                  text: response.text,
                  done: true,
                });
              });
          }
        },
        (error: Error) => {
          client.emit('error', { message: error.message });
        },
      );
    }

    // Forward audio to Deepgram
    session.transcription.sendAudio(Buffer.from(data));
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

  @SubscribeMessage('text-query')
  async handleTextQuery(
    @MessageBody() data: { text: string; model?: string },
    @ConnectedSocket() client: Socket,
  ) {
    const session = this.clientSessions.get(client.id);
    const model = data.model || session?.model || 'gemini-flash';

    client.emit('processing-start', {});

    const response = await this.aiService.generateResponse({
      transcript: data.text,
      screenshotBase64: session?.lastScreenshot || undefined,
      resumeContext: this.ragService.getContext(),
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
    if (data.geminiKey) apiKeys.gemini = data.geminiKey;
    if (data.openaiKey) apiKeys.openai = data.openaiKey;
    if (data.anthropicKey) apiKeys.anthropic = data.anthropicKey;

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
