import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import WebSocket from 'ws';

@Injectable()
export class TranscriptionService {
  private readonly logger = new Logger(TranscriptionService.name);
  private deepgramApiKey: string;

  constructor(private configService: ConfigService) {
    this.deepgramApiKey =
      this.configService.get<string>('DEEPGRAM_API_KEY') || '';
  }

  updateApiKey(key: string) {
    this.deepgramApiKey = key;
    this.logger.log('Deepgram API key updated');
  }

  /**
   * Create a live transcription WebSocket connection to Deepgram.
   * Returns methods to send audio data and close the connection.
   */
  createLiveTranscription(
    onTranscript: (text: string, isFinal: boolean) => void,
    onError: (error: Error) => void,
  ): {
    sendAudio: (data: Buffer) => void;
    close: () => void;
  } {
    if (!this.deepgramApiKey) {
      this.logger.warn('Deepgram API key not set. Transcription unavailable.');
      return {
        sendAudio: () => {},
        close: () => {},
      };
    }

    let ws: WebSocket | null = null;

    try {
      const url =
        'wss://api.deepgram.com/v1/listen' +
        '?model=nova-2&language=en' +
        '&smart_format=true&punctuate=true' +
        '&interim_results=true';

      // @ts-ignore - types mismatch with global WebSocket
      ws = new WebSocket(url, {
        headers: {
          Authorization: `Token ${this.deepgramApiKey}`,
        },
      } as any);

      ws.onopen = () => {
        this.logger.log('Deepgram WebSocket connected');
      };

      ws.onmessage = (event: any) => {
        try {
          const raw =
            typeof event.data === 'string' ? event.data : String(event.data);
          const data = JSON.parse(raw) as {
            channel?: {
              alternatives?: { transcript?: string }[];
            };
            is_final?: boolean;
          };
          const transcript = data.channel?.alternatives?.[0]?.transcript;
          const isFinal = data.is_final ?? false;

          if (transcript && transcript.trim().length > 0) {
            onTranscript(transcript.trim(), isFinal);
          }
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : 'Unknown error';
          this.logger.error(`Failed to parse Deepgram response: ${errMsg}`);
        }
      };

      ws.onerror = () => {
        this.logger.error('Deepgram WebSocket error');
        onError(new Error('Deepgram connection error'));
      };

      ws.onclose = () => {
        this.logger.log('Deepgram WebSocket closed');
      };
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Unknown error';
      this.logger.error(`Failed to create Deepgram connection: ${errMsg}`);
      onError(
        err instanceof Error ? err : new Error('Deepgram connection failed'),
      );
    }

    return {
      sendAudio: (data: Buffer) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data);
        }
      },
      close: () => {
        if (ws) {
          ws.close();
          ws = null;
        }
      },
    };
  }
}
