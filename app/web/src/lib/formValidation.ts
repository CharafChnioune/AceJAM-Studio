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

export function collectPayloadBlockingIssues(validation: unknown, payload: unknown): string[] {
  const seen = new Set<string>();
  const messages: string[] = [];
  const push = (value: unknown) => {
    const text = typeof value === "string" ? value.trim() : "";
    if (!text || seen.has(text)) return;
    seen.add(text);
    messages.push(text);
  };
  const consumeList = (value: unknown) => {
    if (!Array.isArray(value)) return;
    for (const item of value) {
      if (typeof item === "string") {
        push(item);
      } else if (item && typeof item === "object") {
        push((item as Record<string, unknown>).detail);
        push((item as Record<string, unknown>).message);
        push((item as Record<string, unknown>).id);
      }
    }
  };
  if (payload && typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    consumeList(record.payload_gate_blocking_issues);
    consumeList(record.rap_blocking_issues);
  }
  if (validation && typeof validation === "object") {
    const record = validation as Record<string, unknown>;
    consumeList(record.blocking_issues);
    const fieldErrors = record.field_errors;
    if (fieldErrors && typeof fieldErrors === "object") {
      for (const value of Object.values(fieldErrors as Record<string, unknown>)) {
        push(value);
      }
    }
  }
  return messages;
}
