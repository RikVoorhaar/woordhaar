import type { Language, TranslationResult, ApiError } from './types';

const API_HOST = import.meta.env.PUBLIC_API_HOST || '127.0.0.1';
const API_PORT = import.meta.env.PUBLIC_API_PORT || '8000';
const API_BASE = `http://${API_HOST}:${API_PORT}`;

export class TranslationError extends Error {
	constructor(
		message: string,
		public statusCode: number,
		public errorCode?: string
	) {
		super(message);
		this.name = 'TranslationError';
	}
}

export async function translate(
	word: string,
	language: Language
): Promise<TranslationResult> {
	if (!word || !word.trim()) {
		throw new TranslationError('Word cannot be empty', 422, 'VALIDATION_ERROR');
	}

	const url = `${API_BASE}/api/translate`;
	const response = await fetch(url, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ word: word.trim(), language })
	});

	if (!response.ok) {
		let errorDetail = 'Translation service temporarily unavailable';
		let errorCode: string | undefined;

		if (response.status === 422) {
			try {
				const errorData: ApiError = await response.json();
				errorDetail = errorData.detail || errorDetail;
				errorCode = errorData.error_code;
			} catch {
				// Fallback to default message
			}
		} else if (response.status >= 500) {
			errorDetail = 'Translation service temporarily unavailable';
		}

		throw new TranslationError(errorDetail, response.status, errorCode);
	}

	return response.json();
}
