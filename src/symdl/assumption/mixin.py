class AssumptionMixin:
    _implications: dict[str, list[str]] = {
        "is_gaussian": ["is_random"],
    }

    def _inject_facts(self, facts: dict[str, bool]) -> dict[str, bool]:
        closure = dict(facts)
        for fact, value in facts.items():
            if value:
                for implied in self._implications.get(fact, []):
                    closure[implied] = True
        return closure

    def apply_assumptions(self, base):
        facts = {
            "is_random": getattr(base, "is_random", False),
            "is_gaussian": getattr(base, "is_gaussian", False),
        }
        for key, value in self._inject_facts(facts).items():
            setattr(self, key, value)
