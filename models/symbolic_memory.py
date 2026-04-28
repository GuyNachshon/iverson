"""
Symbolic Memory Buffer for ARC-AGI-3.

Stores discovered rules as explicit IF-THEN predicates:
  IF player_adjacent_to(blue) AND key=2 THEN object_removed(blue)

Unlike RSSM latent states, symbolic rules:
  1. Never degrade over time
  2. Can be composed (rule1 AND rule2)
  3. Can be queried ("how do I remove an object?")
  4. Persist perfectly across levels

Also includes:
  - RuleInducer: extracts rules from observed transitions
  - BoredomDetector: triggers diversification when stuck

See full implementation in sandbox at /app/arc_agent_v2/models/symbolic_memory.py
"""
