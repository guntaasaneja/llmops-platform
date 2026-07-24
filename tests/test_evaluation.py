from app.services.cost_tracker import estimate_cost
from app.services.evaluation_service import evaluate_completion
from app.services.hallucination_eval import score_hallucination


def test_estimate_cost_is_nonnegative():
    cost = estimate_cost(prompt_tokens=100, completion_tokens=200)
    assert cost >= 0


def test_hallucination_perfect_match_scores_low():
    reference = "Paris is the capital of France."
    completion = "Paris is the capital of France."
    score = score_hallucination(completion, reference)
    assert score < 0.2


def test_hallucination_unrelated_text_scores_high():
    reference = "Paris is the capital of France."
    completion = "Bananas are a great source of potassium and fiber for athletes."
    score = score_hallucination(completion, reference)
    assert score > 0.6


def test_hallucination_no_reference_returns_neutral_score():
    score = score_hallucination("Some completion text.", None)
    assert score == 0.5


def test_evaluate_completion_returns_expected_shape():
    result = evaluate_completion(
        prompt="What is the capital of France?",
        completion="The capital of France is Paris.",
        reference="Paris is the capital of France.",
    )
    assert 0.0 <= result.relevance_score <= 1.0
    assert 0.0 <= result.hallucination_score <= 1.0
    assert 0.0 <= result.toxicity_score <= 1.0
    assert isinstance(result.overall_pass, bool)
