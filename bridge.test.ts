import { describe, expect, it } from "vitest";
import { applyCompressionResult, buildCompressionPayload } from "./bridge.ts";
import type { AgentMessage, OpenAIMessage } from "./types.ts";

function createAssistantMessage(): AgentMessage {
	return {
		role: "assistant",
		content: [
			{
				type: "toolCall",
				id: "call_1",
				name: "list_records",
				arguments: { limit: 1000 },
			},
		],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-test",
		usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: zeroCost() },
		stopReason: "toolUse",
		timestamp: 1,
	} as AgentMessage;
}

function createToolResult(text: string, toolName = "list_records"): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: "call_1",
		toolName,
		content: [{ type: "text", text }],
		details: { rows: 1000 },
		isError: false,
		timestamp: 2,
	} as AgentMessage;
}

function createUserMessage(text: string): AgentMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: 0 } as AgentMessage;
}

function zeroCost() {
	return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };
}

describe("headroom bridge", () => {
	it("applies compressed toolResult text while preserving Pi metadata", () => {
		const originalText = JSON.stringify(Array.from({ length: 10 }, (_, id) => ({ id, status: "ok" })));
		const messages = [createUserMessage("summarize records"), createAssistantMessage(), createToolResult(originalText)];
		const payload = buildCompressionPayload(messages, 10);
		const compressed = payload.messages.map((message): OpenAIMessage => {
			if (message.role === "tool") {
				return { ...message, content: "[10 records compressed to 2 representative rows]" };
			}
			return message;
		});

		const result = applyCompressionResult(messages, payload.mappings, compressed, { minMessageChars: 10 });

		expect(result.ok).toBe(true);
		if (!result.ok) return;
		const toolResult = result.messages[2] as Extract<AgentMessage, { role: "toolResult" }>;
		expect(toolResult.toolName).toBe("list_records");
		expect(toolResult.details).toEqual({ rows: 1000 });
		expect(toolResult.content).toEqual([{ type: "text", text: "[10 records compressed to 2 representative rows]" }]);
	});

	it("rejects compression when message count changes", () => {
		const messages = [createUserMessage("go"), createAssistantMessage(), createToolResult("large result")];
		const payload = buildCompressionPayload(messages, 5);

		const result = applyCompressionResult(messages, payload.mappings, payload.messages.slice(1), {
			minMessageChars: 5,
		});

		expect(result).toEqual({ ok: false, reason: "message-count-changed" });
	});

	it("sends minimal assistant tool-call context with tool-result candidates", () => {
		const messages = [createUserMessage("exact user intent"), createAssistantMessage(), createToolResult("large result")];

		const payload = buildCompressionPayload(messages, 5);

		expect(payload.messages).toEqual([
			{
				role: "assistant",
				content: null,
				tool_calls: [
					{
						id: "call_1",
						type: "function",
						function: { name: "list_records", arguments: JSON.stringify({ limit: 1000 }) },
					},
				],
			},
			{ role: "tool", content: "large result", tool_call_id: "call_1" },
		]);
		expect(payload.candidateCount).toBe(1);
		expect(payload.mappings).toHaveLength(2);
	});

	it("does not treat small tool results as compression candidates", () => {
		const messages = [createUserMessage("go"), createAssistantMessage(), createToolResult("small")];

		const payload = buildCompressionPayload(messages, 100);

		expect(payload.candidateCount).toBe(0);
	});

	it("sends exact-context tool results with tool names so Headroom can protect them", () => {
		const largeText = "exact file content\n".repeat(200);
		const assistant = createAssistantMessage();
		(assistant as { content: Array<{ name?: string }> }).content[0].name = "read";
		const messages = [createUserMessage("read it"), assistant, createToolResult(largeText, "read")];

		const payload = buildCompressionPayload(messages, 10);

		expect(payload.candidateCount).toBe(1);
		expect(payload.messages[0]).toMatchObject({
			role: "assistant",
			tool_calls: [{ id: "call_1", function: { name: "read" } }],
		});
		expect(payload.messages[1]).toEqual({ role: "tool", content: largeText, tool_call_id: "call_1" });
	});
});
