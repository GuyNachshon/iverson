Dataset Description
-------------------

ARC-AGI-3 is an **Interactive Reasoning Benchmark** designed to measure an AI agent's ability to generalize in novel, unseen environments. Unlike traditional static benchmarks used to evaluate LLMs and reasoning systems, ARC-AGI-3 evaluates frontier AI agent systems on exploration, memory, goal acquisition, and alignment.

Full documentation: [docs.arcprize.org](https://docs.arcprize.org/)

Games (Environments)
--------------------

ARC-AGI-3 consists of **hand-crafted interactive environments** that test abstraction and reasoning. Each game presents a unique challenge that your agent must explore, understand, and solve.

### How Games Work

*   Your agent receives **frames** — JSON objects containing the current game state and metadata.
*   Each frame includes a **grid** (max 64×64) with integer cell values 0–15 representing different states/colors, using a (0,0) top-left coordinate system.
*   Your agent responds with **actions** to interact with the environment.
*   Each game has multiple **levels** of increasing difficulty.
*   A game can be in one of three states: `NOT_FINISHED`, `WIN`, or `GAME_OVER`.

### Available Actions

Agents interact with environments using up to 7 actions:

Action

Description

`RESET`

Start or restart the game

`ACTION1` – `ACTION5`

Simple actions (e.g., move up/down/left/right, interact)

`ACTION6`

Complex action requiring `(x, y)` coordinates

`ACTION7`

Additional simple action

Each game defines which actions are available and what they do. The meaning of actions varies per game — your agent must figure out what each action does through exploration.

### Public Games

A set of public games is available for development and practice at [arcprize.org](https://arcprize.org/arc-agi/3). In addition, public game files are available in the `environment_files` folder on this page.

> **Note:** Competition evaluation uses a **separate, private set of 110 games** that your agent has never seen. Half of these are used for the Public Leaderboard score, and the other half for the Private Leaderboard score.

Scoring
-------

AI agents are scored on **two criteria**:

1.  **Completion** — How many levels did the agent complete in each game?
2.  **Efficiency** — How many actions did the agent take to complete each level, compared to a human baseline?

### Scoring Method

*   For each level completed, the agent's action count is compared to a human baseline (first-time test-testers).
*   **Per-level score** = `min(human_actions / agent_actions, 1.0)`, then **squared** (a raw score of 0.5 becomes 0.25).
*   **Per-game score** = Weighted average of level scores (weighted by level index, 1-indexed).
*   **Total score** = Average of all individual game scores.
*   Final output is a score between **0%–100%**.

ARC-AGI Toolkit
---------------

The [`arc-agi`](https://github.com/arcprize/arc-agi) Python package provides the core toolkit for interacting with ARC-AGI-3 environments.

Building Agents
---------------

The [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) repository provides the framework for building and running agents.

### Agent Architecture

An agent plays ARC-AGI-3 by implementing two core methods:

1.  **`is_done(frames, latest_frame)`** — Decide if the agent should stop playing
2.  **`choose_action(frames, latest_frame)`** — Choose which action to take given the current game state

A **Swarm** orchestrates multiple agent instances across all available games in parallel.

### Agent Lifecycle

1.  Get the list of available games from the API
2.  Open a scorecard (tracks performance)
3.  For each game, `RESET` to start, then take actions based on the agent's strategy
4.  Close the scorecard when all games are complete

Files
-----

*   **ARC-AGI-3-Agents/** - a local copy of the [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) repo.
*   **arc\_agi\_3\_wheels/** - package wheels for the installing [ARC-AGI-3](https://github.com/arcprize/ARC-AGI-3).
*   **environment\_files/** - location of the 25 public game files