'use server';

import { z } from 'zod';
import { localOpenAI, LOCAL_LLM_MODEL } from '@/ai/localClient';

const ContextBlockSchema = z.object({
  title: z.string().min(1),
  content: z.string().min(1),
});

const ContextualHelpInputSchema = z.object({
  question: z.string().min(1).describe('The user’s question.'),
  department: z.string().min(1).describe('The department selected by the user.'),
  user_id: z.string().optional().describe('User identifier for long-term memory.'),
  context: z.array(ContextBlockSchema).optional(),
});
export type ContextualHelpInput = z.infer<typeof ContextualHelpInputSchema>;

const ContextualHelpOutputSchema = z.object({
  response: z.string().describe('The chatbot’s contextually relevant response.'),
  usage: z
    .object({
      promptTokens: z.number().nonnegative().optional(),
      completionTokens: z.number().nonnegative().optional(),
      totalTokens: z.number().nonnegative().optional(),
      systemFingerprint: z.string().optional(),
    })
    .optional(),
});
export type ContextualHelpOutput = z.infer<typeof ContextualHelpOutputSchema>;

export async function getContextualHelp(
  rawInput: ContextualHelpInput
): Promise<ContextualHelpOutput> {
  const input = ContextualHelpInputSchema.parse(rawInput);

  const contextSections = input.context?.length
    ? input.context
        .map((block) => `### ${block.title}\n${block.content}`)
        .join('\n\n')
    : 'No prior context provided. Answer using general knowledge and the department rules.';

  const systemPrompt =
    "You are a helpful assistant for the " + input.department + " department.\n" +
    "Always personalize using LONG-TERM USER MEMORY first, then the CURRENT SESSION history. Keep responses concise and context-aware.\n" +
    "When using memories, weave them naturally (no need to say 'theo trí nhớ dài hạn'). If CONTEXT has entries, summarize the key facts in one friendly sentence; only say 'không có ký ức' if the section is empty.\n\n" +
    "**CONTEXT**\n" +
    contextSections +
    "\n\n" +
    "**RULES:**\n" +
    "1.  **Language:** Detect the user's language and reply ONLY in that language.\n" +
    "2.  **Tone:** Friendly, natural; never state you are an AI.\n" +
    "3.  **Formatting:** Use Markdown (no Markdown tables; if needed, use valid HTML table tags).\n" +
    "4.  **Memory usage:** Use the facts from CONTEXT naturally; if no entries, continue politely without claiming you remember nothing unless explicitly empty.\n" +
    "5.  **Praise response:** If someone praises you, answer briefly and warmly, mentioning anh Hưng Phòng CNTT một cách tự nhiên (không rập khuôn, không dài dòng).\n";
  const messages: { role: 'system' | 'user' | 'assistant'; content: string }[] = [
    { role: 'system', content: systemPrompt },
    {
      role: 'user',
      content:
        "Based on the context above (treat it as authoritative and override any prior knowledge), answer the following question as precisely as possible:\n" +
        input.question,
    },
  ];

  console.log('[contextual-help] messages payload', messages);
  console.log('[contextual-help] user_id', rawInput?.user_id);

  const resp = await localOpenAI.chat.completions.create({
    model: LOCAL_LLM_MODEL,
    messages,
  });

  const text =
    resp.choices?.[0]?.message?.content?.toString() ??
    '[No content returned from local model]';

  const usage = resp.usage
    ? {
        promptTokens: resp.usage.prompt_tokens ?? undefined,
        completionTokens: resp.usage.completion_tokens ?? undefined,
        totalTokens: resp.usage.total_tokens ?? undefined,
        systemFingerprint: resp.system_fingerprint ?? undefined,
      }
    : undefined;

  return ContextualHelpOutputSchema.parse({ response: text, usage });
}
