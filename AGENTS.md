# Repository Guidelines

## Project Structure & Module Organization
This repository is organized by learning artifact rather than by Python package:

- `examples/`: reference notebooks for core NLP topics such as regex, text representation, Word2Vec, and classification.
- `practices/`: lab notebooks (`LS01_...` through `LS05_...`) intended for hands-on work.
- `notes/`: supporting theory and walkthroughs in Markdown.
- `exams/1st_partial/`: exam exercises and mock material.
- `architecture.ipynb`: high-level course notebook at the repository root.

Keep new material in the closest topic folder, and prefer descriptive filenames that match the existing pattern.

## Context strategy

Before coding:
1. Read `docs/index.md`.
2. Identify the topic of the task.
3. Read only the files listed for that topic.
4. Prefer `notes/` files over PDFs.
5. Do not read files in `docs/sources/pdf/` unless explicitly requested.

## Folder roles

- `examples/`: reference notebooks from class or tutorials.
- `practices/`: exercises to complete or improve.
- `exams/`: exam-related exercises.
- `notes/`: processed theoretical explanations in Markdown.
- `docs/sources/pdf/`: original PDF sources.
- `docs/extracted/`: extracted text from PDFs.
- `docs/summaries/`: compressed summaries of PDF theory.

## Build, Test, and Development Commands
There is no central build system in this repository. Use Jupyter locally:

- `jupyter lab`: open and run notebooks interactively.
- `jupyter notebook`: lighter alternative for notebook editing.
- `python -m nbconvert --to notebook --execute practices/LS01_regex.ipynb`: execute a notebook to verify it runs start to finish.
- `python -m nbconvert --to markdown notes/word2vec.md`: useful when checking Markdown export compatibility.

Run commands from the repository root so relative paths inside notebooks stay stable.

## Coding Style & Naming Conventions
Use 4-space indentation in Python cells. Prefer clear, incremental notebook cells over long monolithic blocks. Name new practice notebooks with the `LS##_topic.ipynb` pattern and keep theory notes in lowercase Markdown filenames such as `classifier_evaluation.md`.

When editing notebooks, keep outputs relevant and avoid committing large transient artifacts unless they are part of the lesson.

## Testing Guidelines
There is no automated test suite yet. Treat notebook execution as the validation step: run changed notebooks with `nbconvert --execute` and confirm all cells complete without manual intervention. For Markdown notes, verify headings, code fences, and internal terminology stay consistent with the paired notebook.

## Commit & Pull Request Guidelines
Recent history uses short, descriptive commit messages in Spanish that summarize the instructional change, for example `Actualización de notebooks...` or `Se añade el examen...`. Follow that style: one commit per coherent topic update, with the scope called out plainly.

For pull requests, include a brief summary, list affected folders (for example `examples/` or `exams/1st_partial/`), and attach screenshots only when notebook output or rendered notes changed materially.
