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
    onTranscript: (text: string, isFinal: boolean, speechFinal?: boolean) => void,
    onError: (error: Error) => void,
    onUtteranceEnd?: () => void,
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
    const bufferQueue: Buffer[] = [];
    let isClosing = false;

    try {
      const url =
        'wss://api.deepgram.com/v1/listen' +
        '?model=nova-2&language=en' +
        '&smart_format=true&punctuate=true' +
        '&interim_results=true' +
        '&endpointing=400' +
        '&utterance_end_ms=1000' +
        '&vad_events=true';

      // @ts-ignore - types mismatch with global WebSocket
      ws = new WebSocket(url, {
        headers: {
          Authorization: `Token ${this.deepgramApiKey}`,
        },
      } as any);

      ws.onopen = () => {
        this.logger.log('Deepgram WebSocket connected');
        // CRITICAL: Flush queued chunks in order.
        // Chunk 0 contains the EBML/WebM container header required by Deepgram to decode Opus.
        while (bufferQueue.length > 0) {
          const chunk = bufferQueue.shift();
          if (chunk && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(chunk);
          }
        }
      };

      ws.onmessage = (event: any) => {
        try {
          const raw =
            typeof event.data === 'string' ? event.data : String(event.data);
          const data = JSON.parse(raw) as {
            type?: string;
            channel?: {
              alternatives?: { transcript?: string }[];
            };
            is_final?: boolean;
            speech_final?: boolean;
          };

          if (data.type === 'UtteranceEnd') {
            this.logger.log('[Deepgram] UtteranceEnd received (speaker paused)');
            onUtteranceEnd?.();
            return;
          }

          const transcript = data.channel?.alternatives?.[0]?.transcript;
          const isFinal = data.is_final ?? false;
          const speechFinal = data.speech_final ?? false;

          if (transcript && transcript.trim().length > 0) {
            this.logger.log(
              `[Deepgram STT] "${transcript.trim()}" (isFinal: ${isFinal}, speechFinal: ${speechFinal})`,
            );
            onTranscript(transcript.trim(), isFinal, speechFinal);
          }
        } catch (err: unknown) {
          const errMsg = err instanceof Error ? err.message : 'Unknown error';
          this.logger.error(`Failed to parse Deepgram response: ${errMsg}`);
        }
      };

      ws.onerror = (err: any) => {
        this.logger.error(`Deepgram WebSocket error: ${err?.message || err}`);
        onError(new Error(err?.message || 'Deepgram live connection error'));
      };

      ws.onclose = (event: any) => {
        this.logger.log(
          `Deepgram WebSocket closed (code: ${event.code}, reason: ${event.reason || 'none'})`,
        );
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
        if (isClosing) return;
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(data);
        } else if (!ws || ws.readyState === WebSocket.CONNECTING) {
          if (bufferQueue.length < 100) {
            bufferQueue.push(data);
          }
        }
      },
      close: () => {
        isClosing = true;
        bufferQueue.length = 0;
        if (ws) {
          try {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'CloseStream' }));
            }
            ws.close();
          } catch {
            // ignore
          }
          ws = null;
        }
      },
    };
  }
}
