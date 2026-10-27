import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class RagService {
  private readonly logger = new Logger(RagService.name);
  private resumeContext: string = '';
  private projectContext: string = '';
  private contextDir: string;
  private openrouterKey: string = '';

  constructor(private configService: ConfigService) {
    this.contextDir = path.join(process.cwd(), 'context');
    this.openrouterKey =
      this.configService?.get<string>('OPENROUTER_API_KEY') ||
      process.env.OPENROUTER_API_KEY ||
      '';
    this.loadContextFiles();
  }

  updateOpenRouterKey(key: string): void {
    this.openrouterKey = key;
    this.logger.log('OpenRouter API key updated for RAG reranker');
  }

  /**
   * Load resume and project files from the context directory.
   */
  private loadContextFiles(): void {
    try {
      if (!fs.existsSync(this.contextDir)) {
        fs.mkdirSync(this.contextDir, { recursive: true });
        this.logger.log(`Created context directory: ${this.contextDir}`);

        // Create sample files
        this.createSampleFiles();
        return;
      }

      // Load resume
      const resumePath = path.join(this.contextDir, 'resume.json');
      if (fs.existsSync(resumePath)) {
        const data = fs.readFileSync(resumePath, 'utf-8');
        this.resumeContext = data;
        this.logger.log('Resume loaded from context/resume.json');
      }

      // Load project details
      const projectPath = path.join(this.contextDir, 'projects.json');
      if (fs.existsSync(projectPath)) {
        const data = fs.readFileSync(projectPath, 'utf-8');
        this.projectContext = data;
        this.logger.log('Projects loaded from context/projects.json');
      }

      // Load any additional .txt files
      const files = fs
        .readdirSync(this.contextDir)
        .filter((f) => f.endsWith('.txt'));
      for (const file of files) {
        const content = fs.readFileSync(
          path.join(this.contextDir, file),
          'utf-8',
        );
        this.projectContext += `\n\n--- ${file} ---\n${content}`;
      }
    } catch (err: any) {
      this.logger.error(`Failed to load context files: ${err.message}`);
    }
  }

  private createSampleFiles(): void {
    const sampleResume = {
      name: 'Your Name',
      title: 'Full Stack MERN Developer',
      experience: [
        {
          company: 'Company Name',
          role: 'Full Stack Developer',
          duration: '2023 - Present',
          highlights: [
            'Built scalable REST APIs with Node.js and Express',
            'Developed React frontends with TypeScript',
            'Managed MongoDB databases and Redis caching',
          ],
        },
      ],
      skills: {
        frontend: ['React', 'Next.js', 'TypeScript', 'HTML/CSS'],
        backend: ['Node.js', 'Express', 'NestJS', 'REST APIs'],
        database: ['MongoDB', 'PostgreSQL', 'Redis'],
        tools: ['Git', 'Docker', 'AWS', 'CI/CD'],
      },
      education: 'Your Education',
    };

    fs.writeFileSync(
      path.join(this.contextDir, 'resume.json'),
      JSON.stringify(sampleResume, null, 2),
    );

    const sampleProjects = {
      projects: [
        {
          name: 'E-Commerce Platform',
          description: 'Full-stack e-commerce application',
          tech: ['React', 'Node.js', 'MongoDB', 'Stripe'],
          highlights: [
            'Payment integration',
            'Order management',
            'Admin dashboard',
          ],
        },
      ],
    };

    fs.writeFileSync(
      path.join(this.contextDir, 'projects.json'),
      JSON.stringify(sampleProjects, null, 2),
    );

    this.logger.log('Created sample context files in context/ directory');
  }

  /**
   * Split resume and project context into distinct semantic passages for reranking.
   */
  getContextChunks(): string[] {
    const chunks: string[] = [];

    // Parse resume if it is JSON
    if (this.resumeContext) {
      let parsed = false;
      try {
        const resumeObj = JSON.parse(this.resumeContext);
        parsed = true;

        if (resumeObj.name || resumeObj.title) {
          chunks.push(
            `Candidate Profile: ${resumeObj.name || ''} - ${resumeObj.title || ''}`,
          );
        }

        if (Array.isArray(resumeObj.experience)) {
          for (const exp of resumeObj.experience) {
            const highlights = Array.isArray(exp.highlights)
              ? exp.highlights.join('; ')
              : '';
            chunks.push(
              `Experience at ${exp.company || 'Company'} (${exp.role || 'Role'}, ${exp.duration || ''}): ${highlights}`,
            );
          }
        }

        if (resumeObj.skills) {
          if (typeof resumeObj.skills === 'object' && !Array.isArray(resumeObj.skills)) {
            const skillsStr = Object.entries(resumeObj.skills)
              .map(([cat, sks]) => `${cat}: ${Array.isArray(sks) ? sks.join(', ') : sks}`)
              .join(' | ');
            chunks.push(`Technical Skills: ${skillsStr}`);
          } else {
            chunks.push(`Technical Skills: ${JSON.stringify(resumeObj.skills)}`);
          }
        }

        if (resumeObj.education) {
          chunks.push(`Education: ${JSON.stringify(resumeObj.education)}`);
        }
      } catch {
        // Not JSON
      }

      if (!parsed) {
        const sections = this.resumeContext
          .split(/\n\s*\n/)
          .map((s) => s.trim())
          .filter((s) => s.length > 20);
        chunks.push(...sections);
      }
    }

    // Projects context
    if (this.projectContext) {
      let parsedProjects = false;
      try {
        const projObj = JSON.parse(this.projectContext);
        if (Array.isArray(projObj.projects)) {
          parsedProjects = true;
          for (const p of projObj.projects) {
            const tech = Array.isArray(p.tech) ? p.tech.join(', ') : '';
            const highlights = Array.isArray(p.highlights)
              ? p.highlights.join('; ')
              : '';
            chunks.push(
              `Project: ${p.name || ''} (${tech}): ${p.description || ''}. Highlights: ${highlights}`,
            );
          }
        }
      } catch {
        // Not JSON
      }

      if (!parsedProjects) {
        const projSections = this.projectContext
          .split(/(?=---\s*.+?\s*---)/g)
          .map((s) => s.trim())
          .filter((s) => s.length > 20);
        chunks.push(...projSections);
      }
    }

    return chunks.length > 0 ? chunks : [this.getContext()];
  }

  /**
   * Use OpenRouter's nvidia/nemotron-3.5-rerank-8b:free model to rerank context chunks
   * against the query to provide laser-focused context to the generation LLM.
   */
  async getRelevantContext(query: string, screenshotBase64?: string): Promise<string> {
    const rawChunks = this.getContextChunks();

    if (!this.openrouterKey || rawChunks.length <= 1) {
      return this.getContext();
    }

    try {
      this.logger.log(
        `[Nemotron Rerank] Reranking ${rawChunks.length} documents for query: "${query.substring(0, 50)}..."`,
      );

      const documents: Array<{ text: string }> = rawChunks.map((chunk) => ({
        text: chunk,
      }));

      const payload = {
        model: 'nvidia/nemotron-3.5-rerank-8b:free',
        query,
        documents,
        top_n: Math.min(documents.length, 3),
      };

      const response = await fetch('https://openrouter.ai/api/v1/rerank', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.openrouterKey}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://github.com/RAMJI735/Interview_ai',
          'X-OpenRouter-Title': 'Interview AI',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errText = await response.text();
        this.logger.warn(`[Nemotron Rerank] API responded with status ${response.status}: ${errText}`);
        return this.getContext();
      }

      const data = await response.json();
      if (data?.results && Array.isArray(data.results) && data.results.length > 0) {
        this.logger.log(
          `[Nemotron Rerank] Selected top ${data.results.length} relevant sections (Best score: ${data.results[0]?.relevance_score})`,
        );

        const rerankedChunks = data.results.map(
          (result: { index: number; relevance_score: number }) => {
            const chunk = rawChunks[result.index];
            return `[Relevance: ${(result.relevance_score * 100).toFixed(1)}%] ${chunk}`;
          },
        );

        return (
          `--- RELEVANT CANDIDATE CONTEXT (Reranked by NVIDIA Nemotron 3.5) ---\n` +
          rerankedChunks.join('\n\n')
        );
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Unknown error';
      this.logger.warn(`[Nemotron Rerank] Execution failed: ${errMsg}. Falling back to default context.`);
    }

    return this.getContext();
  }

  /**
   * Get the combined context for the AI prompt.
   */
  getContext(): string {
    let context = '';
    if (this.resumeContext) {
      context += `RESUME:\n${this.resumeContext}\n\n`;
    }
    if (this.projectContext) {
      context += `PROJECT DETAILS:\n${this.projectContext}`;
    }
    return context || 'No resume or project context provided.';
  }

  /**
   * Update resume context at runtime (from file upload).
   */
  updateResumeContext(content: string): void {
    this.resumeContext = content;
    this.logger.log('Resume context updated at runtime');
  }

  /**
   * Reload context from disk.
   */
  reloadContext(): void {
    this.loadContextFiles();
    this.logger.log('Context reloaded from disk');
  }
}
