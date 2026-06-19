export function collectValidationMessages(value: unknown, seen = new Set<string>()): string[] {
  const messages: string[] = [];
  if (!value || typeof value !== "object") return messages;
  if ("message" in value && typeof value.message === "string") {
    const text = value.message.trim();
    if (text && !seen.has(text)) {
      seen.add(text);
      messages.push(text);
    }
  }
  for (const nested of Object.values(value)) {
    messages.push(...collectValidationMessages(nested, seen));
  }
  return messages;
}
