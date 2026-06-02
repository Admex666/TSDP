# GUIDELINES.md

Version: 1.0
Purpose: Define strict behavioral and technical rules for AI Agents contributing to this repository.

---

## 1. General Operating Principles

You MUST:

- Read the full relevant file(s) before modifying them.
- Understand project structure before generating new files.
- Follow existing architectural and stylistic patterns.
- Make the smallest possible change to satisfy the requirement.
- Preserve backward compatibility unless explicitly instructed otherwise.
- Clearly state assumptions if any ambiguity exists.
- Prefer modifying existing code over rewriting entire files.

You MUST NOT:

- Invent APIs, functions, libraries, or configuration options.
- Assume dependencies exist unless verified in project files.
- Remove code unless explicitly instructed.
- Introduce breaking interface changes without explicit approval.
- Hardcode secrets, credentials, or environment-specific values.
- Generate placeholder logic without clearly marking it.

---

## 2. Context Awareness

Before implementing any change:

1. Identify related modules and dependencies.
2. Check for similar existing implementations.
3. Ensure consistency with current patterns.
4. Verify naming conventions.
5. Review configuration files (package.json, requirements.txt, etc.).

If context is insufficient, explicitly state what is missing.

---

## 3. Code Quality Standards

All generated code MUST:

- Compile without errors.
- Contain correct imports.
- Follow existing linting and formatting rules.
- Avoid unused variables and dead code.
- Use explicit error handling for async or external operations.
- Avoid duplication of logic.

Prefer:

- Pure functions where possible.
- Clear variable naming.
- Explicit typing (if applicable).
- Defensive programming for external inputs.

---

## 4. Security Rules (Mandatory)

- Never hardcode secrets.
- Always validate external inputs.
- Prevent injection vulnerabilities.
- Sanitize user-provided data.
- Avoid exposing internal error details.
- Use secure defaults.

If security implications are unclear, explicitly mention potential risks.

---

## 5. Dependency Management

- Only use dependencies already listed in the project.
- If a new dependency is required, justify it clearly.
- Do not assume latest versions without checking project constraints.
- Avoid deprecated APIs.

---

## 6. Testing Requirements

For every logic change:

- Update existing tests if necessary.
- Add tests for new functionality.
- Cover edge cases and failure paths.
- Avoid false-positive tests.
- Ensure deterministic behavior.

If test framework is unknown, inspect project structure before generating tests.

---

## 7. Refactoring Rules

When refactoring:

- Do not change public interfaces unless instructed.
- Ensure no regression is introduced.
- Maintain compatibility.
- Improve clarity without altering behavior.
- Update dependent modules accordingly.

After refactoring, verify:

- Imports remain valid.
- Types and contracts remain intact.
- No circular dependencies are introduced.

---

## 8. Performance Awareness

Avoid:

- Unnecessary loops.
- Repeated heavy computations.
- Excessive API calls.
- Blocking operations in async flows.

Be mindful of scalability and memory usage.

---

## 9. Documentation Standards

- Update README if behavior changes.
- Add concise comments for non-obvious logic.
- Keep comments aligned with implementation.
- Avoid redundant comments.

---

## 10. Self-Verification Checklist (MANDATORY BEFORE FINALIZING)

Before submitting changes, verify:

- [ ]  All imports exist

[ ] Code compiles

[ ] No hardcoded secrets

[ ] Error handling implemented

[ ] Edge cases covered

[ ] Tests updated

[ ] No breaking changes introduced

[ ] No unused code remains

[ ] Follows project conventions

If any item fails, fix it before finalizing.

---

## 11. Handling Uncertainty

If uncertain:

- State uncertainty explicitly.
- Offer alternative approaches if applicable.
- Avoid fabricating solutions.
- Ask for clarification rather than guessing.

---

## 12. Priority Order (If Conflicts Occur)

1. Security
2. Correctness
3. Backward compatibility
4. Consistency with project patterns
5. Performance
6. Code elegance

---

## 13. Agent Output Expectations

When providing code:

- Include only relevant changes.
- Do not repeat unchanged large files.
- Explain non-obvious decisions briefly.
- Clearly separate assumptions from facts.

---

END OF GUIDELINES