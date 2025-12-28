import { Injectable } from '@nestjs/common';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { buildRoadmapPrompt } from '../../utils/index';
import { ParsedRoadmap } from '../../types/index';
import { GoalPreprocessorService } from '../preprocessor/goal-preprocessor.service';

export interface RoadmapResponse {
  roadmap: ParsedRoadmap;
  shouldTriggerCalendar: boolean;
  calendarIntentReason?: string;
}

@Injectable()
export class ChatService {
  private genAI: GoogleGenerativeAI;
  private model;

  constructor(private readonly goalPreprocessor: GoalPreprocessorService) {
    this.genAI = new GoogleGenerativeAI(process.env.GOOGLE_AI_API_KEY!);
    // Use 'gemini-1.5-flash' for speed/cost, or 'gemini-1.5-pro' for better reasoning
    this.model = this.genAI.getGenerativeModel({
      model: 'gemini-1.5-flash', // or 'gemini-1.5-pro'
      generationConfig: {
        temperature: 0.7,
        responseMimeType: 'application/json', // Critical: forces JSON output
      },
      // Optional: Add safety settings if needed
      // safetySettings: [...],
    });
  }

  async generateRoadmap(userMessage: string): Promise<RoadmapResponse> {
    const { goal, known, experienceLevel } = await this.goalPreprocessor.preprocess(userMessage);
    const prompt = buildRoadmapPrompt(goal, known, experienceLevel);

    // Define strict response schema for reliable parsing
    const responseSchema = {
      type: 'object',
      properties: {
        roadmap: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              title: { type: 'string' },
              description: { type: 'string' },
              nodes: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    id: { type: 'string' },
                    title: { type: 'string' },
                    description: { type: 'string' },
                    resources: {
                      type: 'array',
                      items: {
                        type: 'object',
                        properties: {
                          type: { type: 'string' },
                          title: { type: 'string' },
                          link: { type: 'string' },
                          description: { type: 'string' },
                        },
                        required: ['type', 'title', 'link', 'description'],
                      },
                    },
                  },
                  required: ['id', 'title', 'description', 'resources'],
                },
              },
            },
            required: ['id', 'title', 'description', 'nodes'],
          },
        },
        triggerCalendar: { type: 'boolean' },
        calendarIntentReason: { type: ['string', 'null'] },
      },
      required: ['roadmap', 'triggerCalendar'],
    };

    const fullPrompt = `
You are an expert roadmap builder and learning specialist.

Your task is to generate a structured learning roadmap based on the goal below.

Additionally, detect if the user wants calendar integration — only set triggerCalendar to true if they explicitly mention scheduling, reminders, deadlines, calendar events, check-ins, etc.

User's original message: "${userMessage}"

Roadmap goal: ${prompt}

Examples where triggerCalendar = true:
- "Make a 6-week plan with weekly reminders"
- "Add this to my calendar with deadlines"

Examples where triggerCalendar = false:
- "Give me a roadmap to learn TypeScript"
- "How to master backend development"

ROADMAP GUIDELINES:
- Each stage: meaningful title, 2–3 sentence description explaining why it's important
- Each node: detailed 2–3 sentence educational description
- Exactly 3 high-quality resources per node (video, article, project mix)
- Use reputable sources (MDN, official docs, FreeCodeCamp, Traversy Media, etc.)
- Realistic and valid-looking links

Output must be valid JSON matching the schema.
`.trim();

    try {
      const result = await this.model.generateContent({
        contents: [{ role: 'user', parts: [{ text: fullPrompt }] }],
        generationConfig: {
          responseSchema, // Enforces structure
        },
      });

      const response = result.response;
      const text = response.text();

      if (!text) {
        throw new Error('Empty response from Gemini');
      }

      const parsed = JSON.parse(text);

      return {
        roadmap: parsed.roadmap,
        shouldTriggerCalendar: parsed.triggerCalendar,
        calendarIntentReason: parsed.calendarIntentReason ?? undefined,
      };
    } catch (err: any) {
      console.error('Gemini generation failed:', err.message);
      console.error('Raw output (if any):', err?.response?.text?.());

      // Fallback: attempt to extract roadmap array if possible
      try {
        const fallbackMatch = text.match(/\[[\s\S]*\]/);
        if (fallbackMatch) {
          const roadmap = JSON.parse(fallbackMatch[0]);
          return {
            roadmap,
            shouldTriggerCalendar: false,
          };
        }
      } catch (fallbackErr) {
        // ignore
      }

      throw new Error(`Failed to generate or parse roadmap: ${err.message}`);
    }
  }
}