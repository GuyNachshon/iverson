"""
Smart Cell-Select Strategy for ARC-AGI-3.

Reduces the 4,096-position action space to ~15-80 candidates using:
1. ClickAffordanceMap: visual analysis identifies interactive cells
2. SmartActionBudget: manages RHAE-aware action budget
3. ConditionalMechanicsTracker: learns when actions work
4. IrreversibilityDetector: prevents catastrophic mistakes

See v2/models/smart_cell_select.py in sandbox for full implementation.
"""
