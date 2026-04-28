# ARC-AGI-3 Agent: Review of v2.1 and a Proposed v3 Direction

## Executive Summary

The v2.1 architecture is a genuinely strong piece of engineering on roughly half of the ARC-AGI-3 problem — the half about learning *what the agent can do*. It is significantly weaker on the half about learning *what the agent should do*. This document reviews v2.1 honestly, identifies the structural asymmetry between mechanics-learning and goal-acquisition, and proposes a v3 architecture that treats goals as first-class objects with the same engineering depth as the existing symbolic memory.

The core proposal: shift the agent's center of gravity from "RL agent with a world model" to "scientific experimenter that maintains hypotheses about both mechanics and goals, runs controlled experiments using undo, and updates a symbolic posterior from observation." This reframing changes what data you need (a synthetic ARC-like generator becomes Phase 0, not Phase 4), what modules you build (a goal hypothesis module joins symbolic memory as a peer), and what the planner is doing (MCTS over a joint posterior, not CEM over a learned reward).

---

## Part 1: What v2.1 Gets Right

Before the criticism, the architecture has several genuinely strong design choices that should be preserved.

**The Undo-as-Reasoning module is the best idea in v2.1.** Reframing undo from a safety net to a controlled-experiment primitive is the kind of insight that changes what's possible. Most agent architectures in interactive environments can only observe — they cannot intervene while holding other variables constant. ATT gives the agent the ability to do real counterfactual experiments: take action A, observe diff, undo, take action B from the same state, observe diff, compare. This is closer to how scientists work than to how RL agents work, and it is the right model for an environment where the rules must be inferred rather than given. The PROBE / VERIFY / EXPLOIT mode structure is sensible and the cost analysis (12 actions for full mechanics map) is realistic.

**The symbolic memory thesis is correct and probably underweighted.** The claim that "RSSM latent state from Level 1 degrades by Level 6, symbolic rules don't" is a thesis-level claim about how cross-level generalization actually works in this benchmark. If true — and it is plausibly true — it implies that symbolic representations should carry the majority of the cross-level transfer load, and the neural world model is mostly there for fast local prediction within a level. This has design consequences that v2.1 hasn't fully internalized; the architecture treats symbolic and neural components as roughly equal cooperating peers, but the data may show the symbolic side is 80% of the value.

**The action effect tracker and rule inducer correctly handle affordance discovery.** The "what changed when I acted, under what preconditions" loop is exactly the right primitive for typing the scene into agent-controlled, environment-dynamic, and static elements. This is the foundation that all goal reasoning rests on, and v2.1 builds it properly.

**State persistence across levels is correctly scoped.** The decision about what persists (mechanics knowledge, rules, affordances) versus what resets (goals, boredom, exploration phase) reflects a real understanding of how levels relate to each other in this benchmark.

**The action budget analysis is grounded.** Targeting 16-32 actions per level against a ~15-action human baseline puts the agent in the 25-50% RHAE range per level, which is realistic and well-calibrated to the squared efficiency penalty.

These are not small things. v2.1 is significantly better thought through than the typical Kaggle baseline, and the modules that are good are very good.

---

## Part 2: The Structural Asymmetry

The fundamental problem with v2.1 is that it answers two questions with very different levels of investment.

**Question 1: What can I do in this environment?** v2.1 answers this with: a 7.4M parameter CNN+RSSM world model, an action effect tracker, a symbolic rule inducer with explicit predicates, an undo-based experimentation system with three operating modes, a boredom detector, and a smart cell-select probe. This is a serious system.

**Question 2: What is this game asking me to do?** v2.1 answers this with: "Visual saliency — rare colored objects are potential goals," plus "after level completion, learn what winning looks like." That is one heuristic and one nice-to-have.

This is upside-down relative to the benchmark's design. ARC-AGI-3 is *specifically* engineered to make goal acquisition the hard part. The mechanics in any individual game are typically simple; what makes the games hard is that you have to figure out what they want without being told. Knoop's launch transcript says this explicitly — the four required capabilities lead with "explore an unknown environment" and "acquire goals," and the failure modes of frontier AI he calls out (anchoring on the wrong game, latching onto early hypotheses) are goal-inference failures, not mechanics-inference failures.

An agent that is excellent at learning what actions do and weak at knowing what to aim for will efficiently navigate to the wrong objective. Under RHAE's squared penalty, that's catastrophic.

The asymmetry shows up in three concrete places.

The roadmap defers terminal-state feature extraction to Phase 2, retroactive goal labeling is mentioned only briefly, and the synthetic ARC-like generator (which is the only realistic source of pretraining data for goal inference) is Phase 4. These are out of order. Goal acquisition machinery should be Phase 1 alongside the world model, because the world model alone is not enough.

The architecture has no module whose job is to maintain and update a posterior over candidate goals. "Subgoal cycling" is a strategy that consumes goals; it does not produce or refine them. The action selector layers (symbolic rule lookup → goal-directed → CEM → cell-select → exploration) all assume the goal is known by the time they run. The implicit assumption is that visual saliency picks the right goal on the first try, and if it doesn't, the boredom detector eventually triggers a reset. This is fragile.

The CEM planner is downstream of the goal — it plans toward an assumed objective. If the objective is wrong, the planner efficiently executes the wrong thing. There is no mechanism for the planner to express uncertainty about the goal itself and act to reduce that uncertainty.

---

## Part 3: A Diagnostic Frame — How Humans Actually Solve These

Recovering what humans bring to a novel game without instructions sharpens the design requirements. It is worth being precise about this because the v2.1 documentation gestures at "human-like" without decomposing it.

Humans bring a small set of meta-priors about what games are: games have goals, goals are typically inferable from visual salience, things that change when I act are probably the things I control, repeated patterns suggest mechanics, levels usually escalate complexity of an existing rule rather than introduce orthogonal ones. This is closer to a Bayesian prior over game-spaces than over pixels.

Humans take an interventional stance from the first action. Within seconds they are running tiny experiments: press button, see what changes, infer the action's effect. This is the capability that ATT correctly captures.

Humans perceive in objects, not pixels. A blob that moves coherently is a thing. A thing that moves when I press a button is *my* thing. A thing that kills me is a hazard. This typing happens before any goal reasoning. v2.1's saliency-based goal inference is gesturing at this but does not have an explicit object-centric representation as the substrate.

Humans form hypotheses and revise cheaply. A human plays for 30 seconds, says "I think I need to reach the green square," tries it, and abandons the hypothesis instantly when it doesn't pan out. The cost of revision is low because the hypothesis is held explicitly, not encoded implicitly in a long action chain. v2.1 has no explicit hypothesis representation, which is exactly why frontier LLMs fail at these games — their hypotheses are encoded in reasoning chains with their own momentum.

Humans compose abstractions across levels. Level 3 builds on levels 1-2 because the *abstractions* transfer, not because the mechanics are identical. Symbolic memory captures this for mechanics; nothing in v2.1 captures it for goals.

These five capabilities — meta-priors, interventionism, object-centric perception, explicit revisable hypotheses, compositional reuse — are what "human-like" decomposes into for this benchmark. v2.1 implements interventionism well (ATT), object-centric perception partially (saliency, but not full slot decomposition), and the others not at all in any explicit form.

---

## Part 4: What v3 Should Change

The proposal is not to throw out v2.1. The mechanics-learning side is solid and should be preserved. The proposal is to add a goal-acquisition system of comparable engineering depth, restructure the planner to operate over joint mechanics-and-goal uncertainty, and reorganize the roadmap so the data infrastructure that enables all of this is built first, not last.

### 4.1 The Synthetic ARC-Like Generator Moves to Phase 0

This is the highest-leverage change in the proposal and it is also the least glamorous. Before any model training, before any new modules, build a procedural generator that produces games drawn from approximately the same design distribution as Chollet's hand-crafted ARC-AGI-3 games.

The reason this is Phase 0 is that almost everything else depends on it. The world model needs pretraining data that resembles ARC games but isn't the 25 public games themselves (those are too few and partially serve as a validation set). The goal hypothesis module needs labeled goal data, which only exists if you generate games with known goals. The symbolic rule prior needs a corpus of mechanics it can learn the *vocabulary* of. The meta-learned adaptation mechanism needs many distinct task distributions to meta-train over.

The investment is real. The right way to start is to spend a focused week playing all 25 public games yourself, deliberately, with notes. Catalog the design patterns: what kinds of objects appear, what action vocabularies are used, what shapes goals take, how levels escalate. The output of that week is essentially the specification for the generator.

Risks: a generator that produces games systematically different from the real distribution will train priors that don't transfer, and the agent will be back to learning everything from the public games at test time. Mitigations: validate the generator by training a small classifier to distinguish generator output from public games, and iterate the generator until the classifier struggles.

This is also the highest-leverage thing the team could do in the first month, because it produces an asset (a labeled corpus of synthetic ARC-like games) that every downstream module benefits from. Without it, every other improvement is bottlenecked on data scarcity.

### 4.2 A Goal Hypothesis Module as a First-Class Peer to Symbolic Memory

The architectural change is to add a module that maintains an explicit posterior over candidate goals, with the same first-class status as the symbolic memory.

The module holds tuples of the form `(goal_predicate, prior_probability, evidence_for, evidence_against, last_updated)`. Goal predicates are drawn from a learned vocabulary — things like `reach(color=X)`, `remove_all(color=Y)`, `match_pattern(P)`, `align(orientation=O)`, `collect_count(N, color=Z)`. The vocabulary is not hand-coded; it is induced from the synthetic generator's training games where goals are known.

On a new game, the module is initialized with a prior over goal predicates conditioned on the parsed scene (number and types of objects, their spatial arrangement, their colors). This is the "what kinds of goals are plausible here" prior, and it is the closest thing in the architecture to the meta-prior humans bring.

As the agent acts, the posterior updates from three signals. First, observed transitions that move the latent state closer to a candidate goal increase that goal's posterior. Second, terminal states (level completions) confirm or refute goals via retroactive labeling — when a level completes, the diff between the terminal state and earlier states is the goal representation, and that representation gets matched against the candidate vocabulary. Third, ATT-style controlled experiments can be directed specifically at goal disambiguation: if two goal hypotheses predict different outcomes for the same action, taking that action becomes maximally informative.

The action selector's layers are then reordered. The current layering (symbolic rule → goal-directed → CEM → cell-select → exploration) assumes a known goal. The new layering becomes: ATT-probe → joint-uncertainty-driven (act to reduce uncertainty in either mechanics or goals, whichever has higher remaining posterior entropy) → symbolic-rule-execution-toward-likely-goal → planning-toward-confirmed-goal → fallback-exploration. The agent does not commit to executing toward a goal until the goal posterior is sharp enough to justify the commitment.

This handles the failure mode Knoop described directly. Frontier AI latches onto an early hypothesis because there is no mechanism for the hypothesis to be cheaply revised. An explicit posterior with explicit evidence accumulation makes revision the default behavior, not a recovery from failure.

### 4.3 Object-Centric Perception as the Substrate

The current CNN encoder is fine for prediction accuracy but it does not produce representations that make affordance and goal reasoning clean. A slot-attention or similar object-centric module — placed between the CNN and the RSSM — would give every downstream module a much better substrate.

The benefit is not abstract. Affordance discovery becomes "which slots changed when I acted" rather than "which pixels changed and can I cluster them." Goal predicates can be expressed naturally over slots — `reach(slot_with_color_X)` is well-typed; `reach(pixel_region_of_color_X)` is fragile to size and position. Counterfactual queries against the world model become "if I had not acted, would slot 3 have changed?" which is a clean question in a slot representation and a messy one in a pixel representation.

The cost is real: slot attention adds parameters and training complexity, and the synthetic generator becomes more important because slot-based models need diverse training distributions to learn good slot decompositions. This is one of the changes that should be validated against v2.1 with an ablation, not adopted on faith.

### 4.4 Goal Inference Without an LLM, but Pretrained Like One

Calling an external LLM at test time is impossible (Kaggle has internet disabled). Bundling a frontier LLM is impractical (weight budget, runtime). But the *capability* an LLM provides — generating plausible structured goal hypotheses from a parsed scene — is exactly what the agent needs.

The proposal is to train a small dedicated goal-hypothesis model on the synthetic generator. Input: parsed scene description (slots, attributes, observed dynamics from the first K actions). Output: a ranked list of candidate goal predicates with prior probabilities. This is a sequence-to-sequence task on synthetic data where the goal is known. A few-million-parameter transformer trained on millions of generator examples is more than sufficient.

This module is what gives the agent its meta-prior about what games are. It is the substitute for "I have played thousands of games in my life and I have a sense of what these usually want." Without it, the agent has no prior over goals and the posterior starts uniform — which is the same situation as a frontier LLM that has never seen an interactive grid game in pretraining.

### 4.5 Replace CEM with MCTS, and Plan Over Joint Uncertainty

CEM is the wrong planner for this problem and the v2.1 roadmap already concedes this. The correction should be moved earlier.

MCTS over the world model with the goal posterior as the value function is the right shape. Action selection at each tree node maximizes expected goal-progress under the current posterior, with exploration bonuses driven by joint mechanics-and-goal uncertainty. When the goal posterior is sharp, MCTS behaves like standard goal-directed planning. When it is broad, MCTS naturally biases toward information-gathering actions, because actions that disambiguate goals have high expected value across the posterior.

This is the right unification. The agent does not have separate "exploration" and "exploitation" phases; it has a single planner that does the right thing depending on what is uncertain.

### 4.6 ATT Becomes a Tool of the Planner, Not a Separate Module

Undo-as-Reasoning is good, but in v2.1 it sits beside the action selector as a mode-switched layer. In v3 it becomes a primitive available to the planner: at any decision point, the planner can choose to execute an action and undo it as a single composite action with a 2-action cost and a known information yield. The planner decides when this is worth it based on expected information gain about either mechanics or goals.

This matters because the value of an undo probe is goal-dependent. Probing key 4's effect is high-value when you don't know what key 4 does and a candidate goal involves an unknown mechanic. It is low-value when goals are sharp and mechanics are known. The mode-based logic (PROBE / VERIFY / EXPLOIT) approximates this but a unified planner gets it exactly right.

---

## Part 5: Revised Roadmap

The phasing changes substantially.

**Phase 0 (data infrastructure, weeks 1-4):** Play all 25 public games and document the design space. Build the synthetic ARC-like game generator. Validate generator quality with a discriminator. Generate the pretraining corpus. This phase produces no agent improvements directly; it produces the asset everything downstream depends on.

**Phase 1 (foundation, weeks 4-8):** Pretrain the world model on synthetic games. Add slot-attention object-centric perception. Train the goal-hypothesis module on synthetic games where goals are labeled. Port v2.1's symbolic memory and action effect tracker. Replace CEM with MCTS over joint mechanics-and-goal uncertainty.

**Phase 2 (goal acquisition, weeks 8-12):** Implement the explicit goal posterior module. Integrate retroactive goal labeling from terminal states. Wire ATT into the planner as an information-gain primitive. Validate on the 25 public games — target is goal-correct-by-action-20 on at least 80% of public games.

**Phase 3 (cross-level transfer, weeks 12-16):** Ablate the symbolic-vs-neural contribution to cross-level generalization. Tune the persistence policy based on results. Add explicit goal-vocabulary persistence (which goal types appear in this environment) alongside mechanic-rule persistence.

**Phase 4 (meta-learning, weeks 16-20):** Algorithm Distillation on learning histories from synthetic games, training the agent to learn faster within an episode. Test-time adaptation of the world model and goal module via small online updates.

**Phase 5 (competition polish, weeks 20-24):** Kaggle packaging, time budgeting per level, ensemble strategies if multiple promising configurations exist.

The total timeline lands around six months, which fits the November 2 deadline with margin if started by early May. Milestone 1 (June 30) catches Phase 1 completion and a respectable submission; Milestone 2 (September 30) catches Phase 3 with strong cross-level performance.

---

## Part 6: What This Buys, Honestly

This proposal is more work than v2.1's roadmap and the benefits should be stated honestly.

The largest benefit is that the agent becomes capable of recovering from wrong initial goal hypotheses. v2.1 will, in many games, navigate efficiently toward the wrong objective and then thrash when the boredom detector fires. v3 maintains uncertainty about the goal and acts to reduce it. On games where the goal is obvious, v3 is no slower than v2.1 because the posterior sharpens immediately. On games where the goal is ambiguous, v3 is dramatically more efficient because it does not commit prematurely.

The second benefit is that the synthetic generator unblocks every subsequent improvement. Without it, every module is bottlenecked on the 25 public games, which are too few to train anything serious. With it, world model pretraining, goal hypothesis training, and meta-learning all become tractable.

The third benefit is that the architecture has a coherent story for why it works, which matters for both research credibility and debugging. v2.1 is a collection of good modules; v3 is an agent that does scientific experimentation under uncertainty over a structured hypothesis space. When something fails, the failure has a name (wrong prior over goals, insufficient mechanics vocabulary, planner exploiting wrong posterior) and a fix.

The risks are also real. The goal-hypothesis module is the riskiest piece — if the synthetic generator's goal vocabulary doesn't cover the held-out games' goal types, the prior is unhelpful and the agent falls back to retroactive labeling, which is slower. The slot-attention encoder may not learn good slot decompositions on grid-world data without careful tuning. MCTS over a learned world model is harder to get right than CEM and may need significant engineering. None of these are fatal but they are real.

The honest summary: v2.1 is a strong attempt at the easier half of ARC-AGI-3. v3 is a credible attempt at the whole problem. The work to get there is substantial but the work is concentrated in a single high-leverage asset (the synthetic generator) and a single architectural change (explicit goal posteriors), with everything else following from those two decisions.

---

If you want, I can turn any one of these sections into a more detailed spec — particularly the synthetic generator design, the goal predicate vocabulary, or the joint-uncertainty MCTS — those are the three places where the proposal is doing the most work and where the engineering details matter most.
