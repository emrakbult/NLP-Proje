import type {
  ExtractedDocumentPayload,
  HealthPayload,
  MatchResult,
  ModelComparisonPayload,
  SamplePayload
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const isFormData = options?.body instanceof FormData;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: isFormData
      ? options?.headers
      : {
          "Content-Type": "application/json",
          ...(options?.headers ?? {})
        },
    ...options
  });

  if (!response.ok) {
    let message = `İstek ${response.status} durum koduyla başarısız oldu`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      // Yanıt JSON değilse genel hata mesajını koru.
    }
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export function getHealth(): Promise<HealthPayload> {
  return request<HealthPayload>("/health");
}

export function getSample(): Promise<SamplePayload> {
  return request<SamplePayload>("/sample");
}

export function analyzeMatch(payload: SamplePayload): Promise<MatchResult> {
  return request<MatchResult>("/analyze", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function compareModels(payload: SamplePayload): Promise<ModelComparisonPayload> {
  return request<ModelComparisonPayload>("/compare", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function extractText(file: File): Promise<ExtractedDocumentPayload> {
  const formData = new FormData();
  formData.append("file", file);

  return request<ExtractedDocumentPayload>("/extract-text", {
    method: "POST",
    body: formData
  });
}
