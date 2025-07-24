from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult

from .constants import ATTR_FORCE_SAMPLE


class ErrorAwareSampler(Sampler):
    """Sampler that forces sampling on errors."""

    attr_force_sample = ATTR_FORCE_SAMPLE

    def __init__(self, base_sampler: Sampler):
        self.base_sampler = base_sampler

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ):
        # Force sample if error attribute is set
        if attributes and attributes.get(self.attr_force_sample):
            return SamplingResult(
                decision=Decision.RECORD_AND_SAMPLE,
                attributes=attributes,
                trace_state=trace_state,
            )

        # Use base sampler otherwise
        return self.base_sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self):
        return f"ErrorAwareSampler(base={self.base_sampler})"
