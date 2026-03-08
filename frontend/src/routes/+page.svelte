<script lang="ts">
	import { translate, TranslationError } from '$lib/api';
	import type { Language, TranslationResult } from '$lib/types';

	let daWord = $state('');
	let nlWord = $state('');
	let enWord = $state('');

	let loading = $state(false);
	let result = $state<TranslationResult | null>(null);
	let error = $state<string | null>(null);

	async function handleTranslate(language: Language, word: string) {
		if (!word || !word.trim()) {
			error = 'Please enter a word';
			result = null;
			return;
		}

		loading = true;
		error = null;
		result = null;

		try {
			result = await translate(word.trim(), language);
		} catch (e) {
			if (e instanceof TranslationError) {
				error = e.message;
			} else {
				error = 'Translation service temporarily unavailable';
			}
			result = null;
		} finally {
			loading = false;
		}
	}

	function handleDaKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			handleTranslate('da', daWord);
		}
	}

	function handleNlKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			handleTranslate('nl', nlWord);
		}
	}

	function handleEnKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			handleTranslate('en', enWord);
		}
	}
</script>

<div class="container mx-auto p-4 max-w-6xl">
	<h1 class="text-3xl font-bold mb-6 text-center">Woordhaar</h1>

	<!-- Input fields -->
	<div class="flex flex-col sm:flex-row gap-4 mb-6">
		<div class="form-control flex-1">
			<label class="label" for="da-input">
				<span class="label-text">Dansk</span>
			</label>
			<input
				id="da-input"
				type="text"
				bind:value={daWord}
				onkeydown={handleDaKeydown}
				disabled={loading}
				placeholder="Enter a word..."
				class="input input-bordered w-full"
			/>
		</div>

		<div class="form-control flex-1">
			<label class="label" for="nl-input">
				<span class="label-text">Nederlands</span>
			</label>
			<input
				id="nl-input"
				type="text"
				bind:value={nlWord}
				onkeydown={handleNlKeydown}
				disabled={loading}
				placeholder="Enter a word..."
				class="input input-bordered w-full"
			/>
		</div>

		<div class="form-control flex-1">
			<label class="label" for="en-input">
				<span class="label-text">English</span>
			</label>
			<input
				id="en-input"
				type="text"
				bind:value={enWord}
				onkeydown={handleEnKeydown}
				disabled={loading}
				placeholder="Enter a word..."
				class="input input-bordered w-full"
			/>
		</div>
	</div>

	<!-- Loading state -->
	{#if loading}
		<div class="flex justify-center items-center py-8">
			<span class="loading loading-spinner loading-lg"></span>
		</div>
	{/if}

	<!-- Error message -->
	{#if error && !loading}
		<div class="alert alert-error mb-6">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				class="stroke-current shrink-0 h-6 w-6"
				fill="none"
				viewBox="0 0 24 24"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
				/>
			</svg>
			<span>{error}</span>
		</div>
	{/if}

	<!-- Results -->
	{#if result && !loading}
		<div class="space-y-6">
			<!-- Word info -->
			<div class="card bg-base-100 shadow-xl">
				<div class="card-body">
					<h2 class="card-title text-2xl">{result.input_word}</h2>
					{#if result.lemmas.length > 0}
						<p class="text-sm text-base-content/70">
							<strong>Lemmas:</strong> {result.lemmas.join(', ')}
						</p>
					{/if}
					{#if result.etymology}
						<p class="text-sm text-base-content/70">
							<strong>Etymology:</strong> {result.etymology}
						</p>
					{/if}
				</div>
			</div>

			<!-- Cognate cluster -->
			{#if result.cognate_cluster.length > 0}
				<div class="alert alert-info">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						class="stroke-current shrink-0 w-6 h-6"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
						></path>
					</svg>
					<div>
						<h3 class="font-bold">Cognate Cluster</h3>
						<div class="text-sm">{result.cognate_cluster.join(' • ')}</div>
					</div>
				</div>
			{/if}

			<!-- Senses -->
			{#each result.senses as sense, senseIdx}
				<div class="card bg-base-100 shadow-xl">
					<div class="card-body">
						<h3 class="card-title text-lg mb-4">
							Sense {senseIdx + 1}: {sense.source_definition}
						</h3>

						<!-- Translations table -->
						<div class="overflow-x-auto">
							<table class="table table-zebra w-full">
								<thead>
									<tr>
										<th>Language</th>
										<th>Word</th>
										<th>Confidence</th>
										<th>Definition</th>
										<th>Cognate</th>
									</tr>
								</thead>
								<tbody>
									{#each Object.entries(sense.translations) as [lang, translations]}
										{#each translations as translation}
											<tr>
												<td class="font-medium">{lang.toUpperCase()}</td>
												<td class="font-semibold">{translation.word}</td>
												<td>
													<span
														class="badge {translation.confidence === 'high'
															? 'badge-success'
															: translation.confidence === 'medium'
																? 'badge-warning'
																: 'badge-error'}"
													>
														{translation.confidence}
													</span>
												</td>
												<td>{translation.definition || '—'}</td>
												<td>
													{#if translation.is_cognate}
														<span class="badge badge-info">Cognate</span>
													{/if}
												</td>
											</tr>
											{#if translation.notes}
												<tr>
													<td colspan="5" class="text-sm text-base-content/70 italic pl-8">
														Note: {translation.notes}
													</td>
												</tr>
											{/if}
										{/each}
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
