"""Prompt Engineering implementation.

Architecture:
    1. PromptTemplate: Reusable prompt templates with placeholders
    2. PromptTechnique: Various prompting techniques (zero-shot, few-shot, CoT, etc.)
    3. PromptOptimizer: Optimizes prompts based on feedback and evaluation
    4. PromptEvaluator: Evaluates prompt quality and performance

Core concepts:
    - Zero-shot prompting: Minimal instructions, no examples
    - Few-shot prompting: Provide examples in the prompt
    - Chain-of-Thought (CoT): Show reasoning steps
    - Self-Ask: Model asks itself clarifying questions
    - Meta-Prompting: Single prompt for diverse tasks
    - Least to Most: Start general, then add specifics
    - Context Amplification: Add supplementary context
    - Iterative Prompting: Break complex tasks into steps

Args:
    template_name: name of the prompt template
    technique: prompting technique to use
    max_tokens: maximum tokens in generated response
    temperature: sampling temperature for generation
    random_seed: random seed for reproducibility
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PromptTemplate:
    template_id: str
    template_text: str
    placeholders: list[str]
    technique: str = "zero-shot"
    metadata: dict[str, Any] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def render(self, **kwargs) -> str:
        prompt = self.template_text
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            prompt = prompt.replace(placeholder, str(value))
        self._cache = {"rendered_prompt": prompt, "kwargs": kwargs}
        return prompt

    def get_placeholder_names(self) -> list[str]:
        return self.placeholders


@dataclass
class PromptTechnique:
    technique_name: str
    description: str
    examples: list[dict[str, str]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    def apply(self, base_prompt: str, context: str | None = None, examples: list[dict[str, str]] | None = None) -> str:
        if self.technique_name == "zero-shot":
            return base_prompt
        elif self.technique_name == "few-shot":
            if examples:
                example_text = "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples])
                return f"{example_text}\n\nInput: {base_prompt}\nOutput:"
            return base_prompt
        elif self.technique_name == "chain-of-thought":
            return f"{base_prompt}\n\nLet's think step by step:"
        elif self.technique_name == "self-ask":
            return f"{base_prompt}\n\nWhat questions do I need to ask to answer this?"
        elif self.technique_name == "least-to-most":
            return f"First, let's understand the basics: {base_prompt}\n\nNow let's build up to the full solution."
        elif self.technique_name == "meta-prompting":
            return f"General instruction: {base_prompt}\n\nApply this to: {context if context else 'any relevant task'}"
        elif self.technique_name == "context-amplification":
            return f"Context: {context if context else 'N/A'}\n\nTask: {base_prompt}"
        elif self.technique_name == "iterative":
            return f"Step 1: {base_prompt}\n\nContinue step by step."
        else:
            return base_prompt

    def get_technique_params(self) -> dict[str, Any]:
        return self.parameters


@dataclass
class PromptExample:
    example_id: str
    input_text: str
    expected_output: str
    context: str | None = None
    technique: str = "zero-shot"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptEvaluator:
    evaluation_metrics: list[str] = field(default_factory=lambda: ["accuracy", "relevance", "clarity", "completeness"])
    _scores: dict[str, list[float]] = field(default_factory=dict, repr=False)

    def evaluate(self, prompt: str, response: str, expected: str | None = None) -> dict[str, float]:
        scores = {}
        words_prompt = set(prompt.lower().split())
        words_response = set(response.lower().split())
        words_expected = set(expected.lower().split()) if expected else set()

        scores["relevance"] = len(words_prompt.intersection(words_response)) / max(len(words_prompt.union(words_response)), 1)
        scores["clarity"] = min(1.0, len(response.split()) / 50.0)
        scores["completeness"] = len(words_response.intersection(words_expected)) / max(len(words_expected), 1) if expected else 0.5
        scores["accuracy"] = scores["completeness"]

        for metric, score in scores.items():
            if metric not in self._scores:
                self._scores[metric] = []
            self._scores[metric].append(score)

        return scores

    def get_average_scores(self) -> dict[str, float]:
        return {metric: np.mean(scores) if scores else 0.0 for metric, scores in self._scores.items()}

    def reset_scores(self) -> None:
        self._scores = {}


@dataclass
class PromptOptimizer:
    learning_rate: float = 0.01
    optimization_iterations: int = 10
    _best_prompt: str | None = None
    _best_score: float = 0.0
    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def optimize(self, base_prompt: str, evaluator: PromptEvaluator, responses: list[tuple[str, str | None]], technique: PromptTechnique | None = None) -> str:
        current_prompt = base_prompt
        for iteration in range(self.optimization_iterations):
            total_score = 0.0
            for response, expected in responses:
                scores = evaluator.evaluate(current_prompt, response, expected)
                total_score += np.mean(list(scores.values()))

            avg_score = total_score / max(len(responses), 1)
            self._history.append({"iteration": iteration, "prompt": current_prompt, "score": avg_score})

            if avg_score > self._best_score:
                self._best_score = avg_score
                self._best_prompt = current_prompt

            current_prompt = self._suggest_improvement(current_prompt, avg_score)

        return self._best_prompt or base_prompt

    def _suggest_improvement(self, prompt: str, score: float) -> str:
        words = prompt.split()
        if score < 0.5:
            if len(words) < 10:
                return prompt + " Please provide a detailed and specific response."
            else:
                return " ".join(words[:len(words)//2]) + " Be concise and accurate."
        return prompt

    def get_optimization_history(self) -> list[dict[str, Any]]:
        return self._history

    def get_best_prompt(self) -> str | None:
        return self._best_prompt

    def get_best_score(self) -> float:
        return self._best_score


@dataclass
class PromptEngineeringModel:
    model_id: str
    base_model_name: str = "default"
    templates: dict[str, PromptTemplate] = field(default_factory=dict)
    techniques: dict[str, PromptTechnique] = field(default_factory=dict)
    evaluator: PromptEvaluator | None = None
    optimizer: PromptOptimizer | None = None
    _current_prompt: str | None = None
    _current_technique: str = "zero-shot"
    _history: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def _init(self) -> None:
        self.evaluator = PromptEvaluator()
        self.optimizer = PromptOptimizer()
        self._register_default_techniques()

    def _register_default_techniques(self) -> None:
        default_techniques = [
            PromptTechnique("zero-shot", "Direct prompt without examples"),
            PromptTechnique("few-shot", "Prompt with examples", parameters={"n_examples": 3}),
            PromptTechnique("chain-of-thought", "Step-by-step reasoning"),
            PromptTechnique("self-ask", "Self-asking clarifying questions"),
            PromptTechnique("least-to-most", "Start general, then add specifics"),
            PromptTechnique("meta-prompting", "Single prompt for diverse tasks"),
            PromptTechnique("context-amplification", "Add supplementary context"),
            PromptTechnique("iterative", "Break complex tasks into steps"),
        ]
        for technique in default_techniques:
            self.techniques[technique.technique_name] = technique

    def register_template(self, template: PromptTemplate) -> None:
        self.templates[template.template_id] = template

    def set_technique(self, technique_name: str) -> None:
        if technique_name not in self.techniques:
            raise ValueError(f"Unknown technique: {technique_name}")
        self._current_technique = technique_name

    def generate_prompt(self, template_id: str, technique: str | None = None, **kwargs) -> str:
        if not self.templates:
            self._init()
        if template_id not in self.templates:
            raise ValueError(f"Unknown template: {template_id}")
        technique_name = technique or self._current_technique
        template = self.templates[template_id]
        base_prompt = template.render(**kwargs)
        if technique_name in self.techniques:
            technique_obj = self.techniques[technique_name]
            final_prompt = technique_obj.apply(base_prompt, kwargs.get("context"), kwargs.get("examples"))
        else:
            final_prompt = base_prompt
        self._current_prompt = final_prompt
        self._history.append({"template_id": template_id, "technique": technique_name, "prompt": final_prompt})
        return final_prompt

    def evaluate_prompt(self, prompt: str, response: str, expected: str | None = None) -> dict[str, float]:
        if self.evaluator is None:
            self._init()
        return self.evaluator.evaluate(prompt, response, expected)

    def optimize_prompt(self, base_prompt: str, responses: list[tuple[str, str | None]]) -> str:
        if self.optimizer is None:
            self._init()
        return self.optimizer.optimize(base_prompt, self.evaluator, responses)

    def get_history(self) -> list[dict[str, Any]]:
        return self._history

    def get_current_prompt(self) -> str | None:
        return self._current_prompt

    def get_available_techniques(self) -> list[str]:
        if not self.techniques:
            self._init()
        return list(self.techniques.keys())

    def save(self, path: str) -> None:
        import json
        data = {
            "model_id": self.model_id,
            "base_model_name": self.base_model_name,
            "current_technique": self._current_technique,
            "templates": {tid: {"template_id": t.template_id, "template_text": t.template_text, "placeholders": t.placeholders, "technique": t.technique} for tid, t in self.templates.items()},
            "techniques": {tname: {"technique_name": t.technique_name, "description": t.description, "parameters": t.parameters} for tname, t in self.techniques.items()},
            "history": self._history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PromptEngineeringModel":
        import json
        with open(path, "r") as f:
            data = json.load(f)
        obj = cls(model_id=data["model_id"], base_model_name=data.get("base_model_name", "default"))
        obj._current_technique = data.get("current_technique", "zero-shot")
        obj._history = data.get("history", [])
        for tid, tdata in data.get("templates", {}).items():
            obj.templates[tid] = PromptTemplate(template_id=tdata["template_id"], template_text=tdata["template_text"], placeholders=tdata["placeholders"], technique=tdata.get("technique", "zero-shot"))
        for tname, tdata in data.get("techniques", {}).items():
            obj.techniques[tname] = PromptTechnique(technique_name=tdata["technique_name"], description=tdata["description"], parameters=tdata.get("parameters", {}))
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_model_name": self.base_model_name,
            "n_templates": len(self.templates),
            "n_techniques": len(self.techniques),
            "current_technique": self._current_technique,
            "n_history_entries": len(self._history),
        }
