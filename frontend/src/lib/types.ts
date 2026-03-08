export type Language = 'da' | 'nl' | 'en';

export type Confidence = 'high' | 'medium' | 'low';

export interface RankedTranslation {
	word: string;
	language: string;
	confidence: Confidence;
	definition: string | null;
	is_cognate: boolean;
	notes: string | null;
}

export interface TranslationSense {
	source_definition: string;
	translations: Record<string, RankedTranslation[]>; // lang -> ranked list
}

export interface TranslationResult {
	input_word: string;
	input_language: string;
	lemmas: string[];
	senses: TranslationSense[];
	etymology: string | null;
	cognate_cluster: string[];
	processing_time_ms: number;
}

export interface ApiError {
	detail: string;
	error_code: string;
}
