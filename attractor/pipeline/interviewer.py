"""Interviewer interface — human-in-the-loop interaction."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Option:
    """A selectable option in a question."""
    label: str = ""
    key: str = ""  # Accelerator key (e.g., "y", "n", "1")
    description: str = ""


@dataclass
class Question:
    """A question presented to a human reviewer."""
    id: str = ""
    text: str = ""
    options: list[Option] = field(default_factory=list)
    node_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_free_text: bool = False
    timeout_seconds: float = 0  # 0 = no timeout


@dataclass
class Answer:
    """A response from the human reviewer."""
    question_id: str = ""
    selected_label: str = ""
    free_text: str = ""
    timed_out: bool = False


class Interviewer(abc.ABC):
    """Interface for human-in-the-loop interactions."""

    @abc.abstractmethod
    def ask(self, question: Question) -> Answer:
        """Present a question and wait for a response."""
        ...


class AutoApproveInterviewer(Interviewer):
    """Automatically selects the first option (or a configured default)."""

    def __init__(self, default_label: str = ""):
        self._default_label = default_label

    def ask(self, question: Question) -> Answer:
        # Select default label if specified
        if self._default_label:
            for opt in question.options:
                if opt.label == self._default_label:
                    return Answer(
                        question_id=question.id,
                        selected_label=opt.label,
                    )

        # Otherwise select the first option
        if question.options:
            return Answer(
                question_id=question.id,
                selected_label=question.options[0].label,
            )
        return Answer(question_id=question.id, selected_label="approve")


class ConsoleInterviewer(Interviewer):
    """Interactive console-based interviewer."""

    def ask(self, question: Question) -> Answer:
        print(f"\n{'=' * 60}")
        print(f"Question: {question.text}")
        print(f"{'=' * 60}")

        if question.options:
            for i, opt in enumerate(question.options):
                key = opt.key or str(i + 1)
                desc = f" - {opt.description}" if opt.description else ""
                print(f"  [{key}] {opt.label}{desc}")

        if question.allow_free_text:
            print("  (or type a free-text response)")

        while True:
            choice = input("\nYour choice: ").strip()
            if not choice:
                continue

            # Match by key
            for opt in question.options:
                if opt.key and opt.key.lower() == choice.lower():
                    return Answer(
                        question_id=question.id,
                        selected_label=opt.label,
                    )

            # Match by option number
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(question.options):
                    return Answer(
                        question_id=question.id,
                        selected_label=question.options[idx].label,
                    )
            except ValueError:
                pass

            # Match by label substring
            for opt in question.options:
                if opt.label.lower().startswith(choice.lower()):
                    return Answer(
                        question_id=question.id,
                        selected_label=opt.label,
                    )

            # Free text
            if question.allow_free_text:
                return Answer(
                    question_id=question.id,
                    free_text=choice,
                )

            print(f"Invalid choice: {choice!r}. Try again.")


class CallbackInterviewer(Interviewer):
    """Uses a callback function for answers."""

    def __init__(self, callback: Callable[[Question], Answer]):
        self._callback = callback

    def ask(self, question: Question) -> Answer:
        return self._callback(question)


class QueueInterviewer(Interviewer):
    """Uses a pre-loaded queue of answers (for testing)."""

    def __init__(self, answers: list[Answer] | None = None):
        self._answers = list(answers or [])
        self._index = 0

    def add_answer(self, answer: Answer) -> None:
        self._answers.append(answer)

    def ask(self, question: Question) -> Answer:
        if self._index < len(self._answers):
            answer = self._answers[self._index]
            answer.question_id = question.id
            self._index += 1
            return answer
        # Default to first option
        if question.options:
            return Answer(
                question_id=question.id,
                selected_label=question.options[0].label,
            )
        return Answer(question_id=question.id, timed_out=True)


class RecordingInterviewer(Interviewer):
    """Wraps another interviewer and records all Q&A pairs."""

    def __init__(self, inner: Interviewer):
        self._inner = inner
        self._log: list[tuple[Question, Answer]] = []

    def ask(self, question: Question) -> Answer:
        answer = self._inner.ask(question)
        self._log.append((question, answer))
        return answer

    @property
    def log(self) -> list[tuple[Question, Answer]]:
        return list(self._log)
