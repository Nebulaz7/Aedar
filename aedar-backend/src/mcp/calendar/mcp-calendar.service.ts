import { Injectable, Logger } from '@nestjs/common';
import { google } from 'googleapis';
import { ParsedRoadmap } from '../../../types/index';

interface CalendarEvent {
  summary: string;
  description: string;
  start: { dateTime: string };
  end: { dateTime: string };
  reminders?: { useDefault: boolean };
}

@Injectable()
export class McpCalendarService {
  private readonly logger = new Logger(McpCalendarService.name);

  // In real app: fetch from DB using userId
  private mockTokens = new Map<string, any>(); // userId → { access_token, refresh_token, expiry_date }

  async addRoadmapToCalendar(
    userId: string,
    roadmap: ParsedRoadmap,
    options: {
      calendarId?: string;
      startDate?: string; // ISO string, e.g., '2026-01-05'
      dryRun?: boolean;
    } = {},
  ) {
    const { calendarId = 'primary', startDate, dryRun = false } = options;

    try {
      const oauth2Client = this.getAuthClient(userId);
      if (!oauth2Client.credentials.access_token) {
        return { success: false, error: 'No valid access token – calendar not connected' };
      }

      const calendar = google.calendar({ version: 'v3', auth: oauth2Client });
      const events = this.generateEventsFromRoadmap(roadmap, startDate);

      if (dryRun) {
        return {
          success: true,
          dryRun: true,
          eventCount: events.length,
          preview: events.slice(0, 6).map(e => ({
            summary: e.summary,
            date: e.start.dateTime.split('T')[0],
          })),
          message: 'Preview ready. Confirm to create events.',
        };
      }

      const created = [];
      for (const event of events) {
        const res = await calendar.events.insert({
          calendarId,
          requestBody: event,
        });
        created.push({
          id: res.data.id,
          htmlLink: res.data.htmlLink,
          summary: res.data.summary,
        });
      }

      this.logger.log(`Created ${created.length} calendar events for user ${userId}`);
      return {
        success: true,
        eventCount: created.length,
        events: created,
        message: 'Roadmap added to your Google Calendar!',
      };
    } catch (error: any) {
      this.logger.error('Calendar MCP failed:', error.message);
      return {
        success: false,
        error: error.message.includes('invalid_grant')
          ? 'Calendar access expired – please reconnect'
          : error.message || 'Failed to add to calendar',
      };
    }
  }

  private getAuthClient(userId: string) {
    const tokens = this.mockTokens.get(userId);
    if (!tokens) throw new Error('No tokens');

    const oauth2Client = new google.auth.OAuth2(
      process.env.GOOGLE_CLIENT_ID!,
      process.env.GOOGLE_CLIENT_SECRET!,
      `${process.env.BACKEND_URL}/mcp/calendar/callback`,
    );

    oauth2Client.setCredentials(tokens);
    return oauth2Client;
  }

  private generateEventsFromRoadmap(roadmap: ParsedRoadmap, startDateIso?: string): CalendarEvent[] {
    const events: CalendarEvent[] = [];
    let currentDate = startDateIso ? new Date(startDateIso) : new Date();

    for (const stage of roadmap) {
      // Stage kickoff event
      events.push({
        summary: `🚀 Start: ${stage.title}`,
        description: stage.description,
        start: { dateTime: new Date(currentDate).toISOString() },
        end: { dateTime: new Date(currentDate.getTime() + 120 * 60 * 1000).toISOString() }, // 2 hours
        reminders: { useDefault: true },
      });

      // One event per node (topic)
      for (const node of stage.nodes) {
        const nodeDate = new Date(currentDate);
        nodeDate.setDate(nodeDate.getDate() + Math.floor(Math.random() * 5) + 1); // spread out

        events.push({
          summary: `📚 ${node.title}`,
          description: `${node.description}\n\nResources:\n${node.resources.map(r => `• ${r.title}: ${r.link}`).join('\n')}`,
          start: { dateTime: nodeDate.toISOString() },
          end: { dateTime: new Date(nodeDate.getTime() + 90 * 60 * 1000).toISOString() }, // 1.5 hours
          reminders: { useDefault: true },
        });
      }

      // Move to next stage (approx 1–2 weeks later)
      currentDate.setDate(currentDate.getDate() + 10 + stage.nodes.length);
    }

    return events;
  }

  // For OAuth flow – store tokens after user connects
  storeTokens(userId: string, tokens: any) {
    this.mockTokens.set(userId, tokens);
    this.logger.log(`Stored calendar tokens for user ${userId}`);
  }
}