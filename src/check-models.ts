
import { GoogleGenerativeAI } from '@google/generative-ai';
import * as dotenv from 'dotenv';
import * as path from 'path';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });

async function listModels() {
  const apiKey = process.env.GEMINI_API_KEY;
  console.log('API Key loaded:', apiKey ? 'YES (Starts with ' + apiKey.substring(0, 4) + ')' : 'NO');
  
  if (!apiKey) {
    console.error('GEMINI_API_KEY not found in .env');
    return;
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  try {
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' }); 
    // Just using the client to list models if possible, but the SDK structure is usually:
    // actually, listing models might require specific call or just try/catch generic.
    // The error says "Call ListModels". In REST that's a specific endpoint. 
    // In SDK, it might not be directly exposed easily on the main class in older versions, 
    // but let's try to just run a simple generateContent with 'gemini-pro' to see if baseline works.
    
    // Test 1: Standard gemini-2.5-flash
    console.log('Testing gemini-2.5-flash (Standard)...');
    const start1 = Date.now();
    try {
        const m = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
        const result = await m.generateContent('Hello');
        console.log(`✅ Success: Standard took ${Date.now() - start1}ms`);
    } catch (e: any) {
        console.log(`❌ Failed Standard: ${e.message}`);
    }

    // Test 2: gemini-2.5-flash with thinkingBudget = 0
    console.log('Testing gemini-2.5-flash (No Thinking)...');
    const start2 = Date.now();
    try {
        const m = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
        const result = await m.generateContent({
            contents: [{ role: 'user', parts: [{ text: 'Hello' }] }],
            generationConfig: {
                // @ts-ignore
                thinkingConfig: {
                    thinkingBudget: 0,
                },
            },
        });
        console.log(`✅ Success: No Thinking took ${Date.now() - start2}ms`);
    } catch (e: any) {
        console.log(`❌ Failed No Thinking: ${e.message}`);
    }

  } catch (error) {
    console.error('Error:', error);
  }
}

listModels();
