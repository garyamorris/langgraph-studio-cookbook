# Transcript

Narration start times (mm:ss) on the video timeline; text is the exact script.

## Scene A - 00:03

LangGraph Deep Dive is a workbench of twenty-three LangGraph examples. Each one solves a concrete problem, from routing support tickets to approving loans, and all of them load into one local server. This video shows how they actually run.

## Scene B - 00:21

Every example is built from the same four ideas. State is a set of typed fields that every step can read. A node is a plain function that reads that state and returns an update. An edge says what runs next, and a conditional edge lets the program decide at run time. Open any example in Studio, press Submit on a blank form, and a ready-made case runs, so you never need to know the data first.

## Scene C - 00:48

Here is a real run, calling a real model. A support ticket says the plug smells of burning. The load ticket node reads it, and triage asks the model to classify it. Then a conditional edge picks exactly one of six next steps, and this one is escalate, not a specialist. Every step wrote a checkpoint, so the whole run can be inspected, or replayed, later.

## Scene D - 01:13

Some work should run in parallel, but you only know how much once you have planned it. In the report orchestrator, one node plans the sections, and the Send API launches a separate write section task for each of them at the same time. A reducer then merges the drafts into one list, and the graph carries on to write the summary, and check every number against the data.

## Scene E - 01:38

Loops are simply edges that point backwards. Here a writer drafts a product announcement. A rule checker written in code, and a critic model, review it side by side, and a judge decides whether to pass, or go round again. This brief is impossible on purpose: sixteen words of required facts, in a ten word limit. So the loop runs to its cap of three drafts, and stops, honestly labelled best effort.

## Scene F - 02:04

Some decisions need a person. This loan screener asks the OpenAI Decisions API for calibrated probabilities, then applies policy in plain code. For a borderline file, the underwriter node calls interrupt. The whole state is saved in a checkpoint under the thread I D, and nothing runs until someone answers. A Command carrying a resume value wakes the same thread exactly where it stopped, and the decision flows on to the letter.

## Scene G - 02:31

Streaming lets you watch, instead of wait. The review digest loads the reviews, runs three analysis stages inside a subgraph, and then writes the summary. While it works, the server streams every event as it happens: tasks starting, checkpoints saved, custom progress messages from inside the nodes, and the digest itself, token by token. Everything on this screen is drawn from that one stream.

## Scene H - 02:57

Memory comes in two kinds. Short term memory is the thread: the checkpointer keeps the conversation, and when it grows too long, the graph folds the old messages into a summary, while pinning facts like an order number. Long term memory is the Store. The memory concierge saves notes about a customer, and a later conversation, on a different thread, recalls them with semantic search.

## Scene I - 03:21

LangGraph Studio shows all of this. Point it at the local server, and every example appears as a graph. Choose a ready-made case from the dropdown, press Submit, and the graph follows each step as it runs. Then click any step to inspect exactly what it read and wrote, including what the model returned. Or go back to any checkpoint, re-run from there, and fork the thread to take a different turn.

## Scene J - 03:48

Twenty-three graphs, ready-made cases for everything that needs data, and an offline mode, so the whole test suite runs without a key. Start the server with U V run langgraph dev, pick a case, and watch it run.
