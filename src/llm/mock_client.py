from dataclasses import dataclass
import json


@dataclass
class MockLLMClient:
    """
    Simple fallback so the UI works without Ollama.
    """

    def generate(self, prompt: str, *, model_name: str = "mock") -> str:
        # Keep the response deterministic so it is easy to test.
        # This is a placeholder until we wire the real Ollama client in.
        if "Return ONLY valid JSON" in prompt and '"proceed"' in prompt and '"questions"' in prompt:
            # Very small simulation of the clarify-first step.
            # We look for a few key details and decide whether we can proceed.
            marker_ans = "User answers (may be empty):"
            answers = ""
            if marker_ans in prompt:
                answers = prompt.split(marker_ans, 1)[1].strip()

            marker_notes = "Raw notes:"
            raw_notes = ""
            if marker_notes in prompt:
                # Take everything after "Raw notes:" up to the marker for user answers (if present).
                if marker_ans in prompt:
                    raw_notes = prompt.split(marker_notes, 1)[1].split(marker_ans, 1)[0].strip()
                else:
                    raw_notes = prompt.split(marker_notes, 1)[1].strip()

            channel = "Email"
            if "Channel:" in prompt:
                try:
                    channel = prompt.split("Channel:", 1)[1].splitlines()[0].strip()
                except Exception:
                    channel = "Email"

            purpose = ""
            if "Purpose:" in prompt:
                try:
                    purpose = prompt.split("Purpose:", 1)[1].splitlines()[0].strip()
                except Exception:
                    purpose = ""

            answers_l = (answers or "").lower()
            notes_l = (raw_notes or "").lower()
            purpose_l = (purpose or "").lower()

            wants_extension = ("extension" in notes_l) or ("extension" in purpose_l) or ("request" in purpose_l and "deadline" in notes_l)
            mentions_deposit = ("deposit" in notes_l) or ("tuition" in notes_l) or ("fee" in notes_l) or ("deposit" in purpose_l)
            mentions_university = ("university" in notes_l) or ("college" in notes_l) or ("program" in notes_l) or ("department" in notes_l)

            has_deadline = any(k in answers_l for k in ["deadline", "date", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "am", "pm"])
            has_amount = any(k in answers_l for k in ["amount", "deposit", "tuition", "fee", "$", "usd", "inr", "eur", "gbp"])
            has_university = any(k in answers_l for k in ["university", "college", "program", "department"])

            # Decide what is critical based on context.
            critical_ok = True
            if wants_extension and not has_deadline:
                critical_ok = False
            if mentions_deposit and not has_amount:
                critical_ok = False
            if mentions_university and not has_university:
                critical_ok = False

            if critical_ok:
                # Proceed, but still ask 1-3 optional questions to improve quality.
                if channel.lower() == "whatsapp":
                    questions = [
                        "Do you want it to sound more formal or more casual?",
                        "Any specific reference number or student ID you want included?",
                    ]
                elif channel.lower() in ["microsoft teams", "teams"]:
                    questions = [
                        "Should the message be short (2-3 sentences) or slightly detailed?",
                        "Do you want to include a proposed time/date for next steps?",
                    ]
                else:
                    questions = [
                        "What exact greeting should I use (e.g., “Dear Professor Smith” or just “Hello”)?",
                        "Any preferred sign-off (e.g., “Sincerely” vs “Best regards”)?",
                    ]
                return json.dumps({"proceed": True, "questions": questions[:3]})

            # Not ready: ask critical questions first.
            if channel.lower() == "whatsapp":
                questions = []
                if wants_extension and not has_deadline:
                    questions.append("What is the exact deadline/date you’re requesting (day + time if possible)?")
                if mentions_university and not has_university:
                    questions.append("Which university/program is this for?")
                if mentions_deposit and not has_amount:
                    questions.append("What is the deposit amount (and currency) involved?")
                if not questions:
                    questions = ["What exactly do you want them to do (approve/reschedule/confirm)?"]
            elif channel.lower() in ["microsoft teams", "teams"]:
                questions = []
                if wants_extension and not has_deadline:
                    questions.append("What exact date/time should the message reference as the new deadline?")
                if mentions_university and not has_university:
                    questions.append("Which university/program should I mention?")
                if mentions_deposit and not has_amount:
                    questions.append("What deposit/fee amount is involved?")
                if not questions:
                    questions = ["What outcome do you want (approve/confirm/update)?"]
            else:
                questions = []
                if wants_extension and not has_deadline:
                    questions.append("What exact deadline/date are you requesting an extension for (include time if relevant)?")
                if mentions_university and not has_university:
                    questions.append("Which university/program is this for (and department if known)?")
                if mentions_deposit and not has_amount:
                    questions.append("What is the deposit/fee amount (include currency)?")
                if wants_extension and has_deadline and mentions_deposit and has_amount and not has_university:
                    questions.append("Who should receive the message (name/title) if you know it?")
                if not questions:
                    questions = ["Who is the recipient and what is the exact outcome you want?"]

            return json.dumps({"proceed": False, "questions": questions[:6]})

        if "strict evaluator (judge)" in prompt.lower() and '"best_index"' in prompt and "candidates" in prompt:
            # Parse candidates from the judge prompt.
            # Expected format includes lines like: "Index 0" then text.
            import re

            # Find all "Index N" occurrences.
            candidates: list[tuple[int, str]] = []
            for match in re.finditer(r"Index\s+(\d+)\s*\n", prompt):
                idx = int(match.group(1))
                start = match.end()
                # Find next "Index " marker.
                next_m = re.search(r"Index\s+(\d+)\s*\n", prompt[start:])
                end = start + (next_m.start() if next_m else len(prompt) - start)
                text = prompt[start:end].strip()
                candidates.append((idx, text))

            # Simple heuristic for mock judge: prefer correct subject presence for email.
            channel = "email"
            m_ch = re.search(r"Channel:\s*(.+)\n", prompt)
            if m_ch:
                channel = m_ch.group(1).strip().lower()
            include_subject = "true" in prompt.lower().split("include subject line (email only):", 1)[-1].splitlines()[0].lower() if "include subject line (email only)" in prompt.lower() else False

            def score_text(t: str) -> float:
                t_l = t.lower()
                has_subject = "subject:" in t_l
                has_closing = any(x in t_l for x in ["sincerely", "regards", "thank you", "yours"])
                has_next = any(x in t_l for x in ["please", "could you", "i would appreciate", "would you be able"])
                s = 0.0
                if channel == "email":
                    s += 1.0 if (has_subject == include_subject) else (-0.5)
                else:
                    s += 1.0 if not has_subject else -1.0
                s += 0.5 if has_closing else -0.2
                s += 0.5 if has_next else -0.2
                return s

            scored = []
            best_idx = None
            best_score = None
            for idx, text in candidates:
                sc = score_text(text)
                if best_score is None or sc > best_score:
                    best_score = sc
                    best_idx = idx
                scored.append(
                    {
                        "index": idx,
                        "overall_score": float(sc),
                        "intent_score": 0.0,
                        "tone_score": 0.0,
                        "channel_score": 0.0,
                        "hallucination_score": 0.0,
                    }
                )

            if best_idx is None:
                best_idx = 0

            return json.dumps({"best_index": best_idx, "candidates": scored})

        if "Output the polished draft only" in prompt:
            if "Channel: WhatsApp" in prompt:
                return "Hi! Thanks for your time. Could you please share an update regarding the matter? If you need any extra details, I can provide them. Thank you!"
            if "Channel: Microsoft Teams" in prompt:
                return "Hi,\n\nThanks for your time. I am requesting an update on the matter below. If you need any additional details, I can share them.\n\nBest regards,\n[Your Name]"
            return "Subject: Polite Update\n\nHi,\n\nThanks for your time. I am writing to request an update regarding the matter below. I appreciate your consideration, and I will provide any additional details if needed.\n\nSincerely,\n[Your Name]"

        if "Output the final edited version only" in prompt:
            if "Channel: WhatsApp" in prompt:
                return "Hi! Thanks for your time. Here is my request for an update on the matter. If you need more details, I can share them. Thank you!"
            if "Channel: Microsoft Teams" in prompt:
                return "Hi,\n\nI am requesting an update on the matter below. Thank you for your consideration. If you need any additional details, I can provide them.\n\nBest regards,\n[Your Name]"
            return "Subject: Polite Update\n\nHi,\n\nI am writing to request an update regarding the matter below. Thank you for your consideration. If you need any additional information, I can provide it.\n\nSincerely,\n[Your Name]"

        return "OK"

    def generate_with_options(self, prompt: str, *, options: dict, model_name: str = "mock") -> str:
        # Ignore options for the mock; we just want the workflow to run.
        return self.generate(prompt, model_name=model_name)

