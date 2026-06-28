from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


TOKENS_PER_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class ModelTokenPricing:
    input_per_million: Decimal
    cached_input_per_million: Decimal
    output_per_million: Decimal


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int

    @property
    def billable_uncached_input_tokens(self) -> int:
        return max(self.input_tokens - self.cached_input_tokens, 0)


@dataclass(frozen=True)
class ApiCallCost:
    model: str
    usage: TokenUsage
    input_cost_usd: Decimal
    cached_input_cost_usd: Decimal
    output_cost_usd: Decimal

    @property
    def total_cost_usd(self) -> Decimal:
        return self.input_cost_usd + self.cached_input_cost_usd + self.output_cost_usd


STANDARD_TEXT_PRICING: dict[str, ModelTokenPricing] = {
    "gpt-5.4": ModelTokenPricing(
        input_per_million=Decimal("2.50"),
        cached_input_per_million=Decimal("0.25"),
        output_per_million=Decimal("15.00"),
    ),
    "gpt-5.4-mini": ModelTokenPricing(
        input_per_million=Decimal("0.75"),
        cached_input_per_million=Decimal("0.075"),
        output_per_million=Decimal("4.50"),
    ),
    "gpt-5.4-nano": ModelTokenPricing(
        input_per_million=Decimal("0.20"),
        cached_input_per_million=Decimal("0.02"),
        output_per_million=Decimal("1.25"),
    ),
    "gpt-5.5": ModelTokenPricing(
        input_per_million=Decimal("5.00"),
        cached_input_per_million=Decimal("0.50"),
        output_per_million=Decimal("30.00"),
    ),
}


def calculate_text_call_cost(model: str, usage: TokenUsage) -> ApiCallCost:
    try:
        pricing = STANDARD_TEXT_PRICING[model]
    except KeyError as exc:
        raise PricingError(f"No standard text pricing configured for model: {model}") from exc

    input_cost = (
        Decimal(usage.billable_uncached_input_tokens)
        * pricing.input_per_million
        / TOKENS_PER_MILLION
    )
    cached_input_cost = (
        Decimal(usage.cached_input_tokens)
        * pricing.cached_input_per_million
        / TOKENS_PER_MILLION
    )
    output_cost = (
        Decimal(usage.output_tokens) * pricing.output_per_million / TOKENS_PER_MILLION
    )

    return ApiCallCost(
        model=model,
        usage=usage,
        input_cost_usd=input_cost,
        cached_input_cost_usd=cached_input_cost,
        output_cost_usd=output_cost,
    )


def token_usage_from_response(response: object) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        raise PricingError("OpenAI response did not include token usage.")

    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    input_details = getattr(usage, "input_tokens_details", None)
    cached_input_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
    )


class PricingError(RuntimeError):
    """Raised when API call cost cannot be calculated from exact usage."""
