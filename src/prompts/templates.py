DRAFT_PROMPT_TEMPLATE = """
You are an expert writing assistant.

Task:
Rewrite the user's raw notes into a polished {purpose} in a {tone} tone for the following audience and channel.

Audience: {audience}
Channel: {channel}
Purpose: {purpose}
Tone: {tone}
Include a subject line (email only): {include_subject}

Word count requirement:
{word_count_block}

Raw notes:
{raw_notes}

Requirements:
- Output the polished draft only (no analysis).
- If include_subject is true, include a concise subject line at the top.
- Use channel-appropriate formatting:
  - Email: include a greeting and a closing/sign-off.
  - WhatsApp: keep it short and natural; no subject line.
  - Microsoft Teams: professional but concise; no subject line.
- Increase variety: write naturally and avoid sounding like the same template every time.
- Keep it coherent and professional.
""".strip()


FINALIZE_PROMPT_TEMPLATE = """
You are a meticulous editor.

Given:
1) A draft message
2) A self-check rubric result

Rewrite the draft to fix issues found by the rubric.

Rubric notes (may include missing/unclear items):
{rubric_notes}

Draft:
{draft}

Word count requirement:
{word_count_block}

Output:
- Output the final edited version only (no headings).
- Use the same channel formatting rules as the draft step.
- Keep wording distinct from the draft while preserving meaning.
""".strip()


CLARIFY_PROMPT_TEMPLATE = """
You are a careful application-writing assistant.

Goal:
Before drafting, identify what information is missing from the user's notes so the final message is specific and correct.

Return ONLY valid JSON with this schema:
{{
  "proceed": boolean,
  "questions": [string, ...]
}}

Rules:
- Always return 2 to 6 questions (questions must not be empty).
- Determine which questions are "critical" (needed for correctness) vs "optional" (to improve quality).
- If any critical information is missing or unclear in (raw notes + user answers), set proceed=false and put the critical questions first.
- If all critical information is present, set proceed=true and include 1 to 3 optional follow-up questions to improve the final message.
- Questions must be concrete (e.g., ask for exact dates, names, deadlines, requested outcome, amounts).
- Do not ask about things the user already included or already clearly answered.

Context:
Channel: {channel}
Purpose: {purpose}
Audience: {audience}
Tone: {tone}

Raw notes:
{raw_notes}

User answers (may be empty):
{user_answers}
""".strip()


JUDGE_DRAFT_PROMPT_TEMPLATE = """
You are a strict evaluator (judge) for message quality.

You will be given:
- The user request context (notes + answers)
- Channel and formatting constraints (email vs WhatsApp vs Teams)
- A candidate drafted message

Return ONLY valid JSON with this schema:
{{
  "best_index": integer,
  "candidates": [
    {{
      "index": integer,
      "overall_score": number,
      "intent_score": number,
      "tone_score": number,
      "channel_score": number,
      "hallucination_score": number
    }}
  ]
}}

Scoring rules:
- intent_score: higher if the draft includes critical details the user provided (deadline/date, deposit amount, university/program) without inventing missing details
- tone_score: higher if wording matches the requested tone and style preset
- channel_score: higher if the draft follows channel rules (e.g., no "Subject:" for WhatsApp/Teams; include subject when required for email)
- hallucination_score: higher if the draft does NOT add details that were not provided
- overall_score is a weighted sum of the above (you choose weights, but keep it reasonable)

Context:
Channel: {channel}
Purpose: {purpose}
Audience: {audience}
Tone: {tone}
Style preset: {style_preset}
Word size target: {word_size}
Include subject line (email only): {include_subject}
Requested finalize: {finalize_requested}

Raw notes:
{raw_notes}

User answers:
{user_answers}

Candidates:
{candidates}
""".strip()


