# Token Minimization Rules

To minimize token usage and keep contexts as small as possible, follow these guidelines strictly:

1. **Concise Communication**:
   - Write extremely brief, direct, and concise responses.
   - Do not write verbose explanations, intros, or summaries unless explicitly asked.

2. **Efficient File Operations**:
   - Only view/read specific line ranges of files when necessary; avoid reading entire large files.
   - Do not load files into context unless they are relevant to the immediate task.

3. **Compact Command Execution**:
   - When running commands, avoid generating massive stdout/stderr. Use piping or pagination filters (e.g., `head`, `tail`, `grep`) where appropriate to avoid dumping thousands of lines of output.

4. **Minimized Planning**:
   - Keep implementation plans and checklists as short and concise as possible.
   - Do not write plans for small or simple tweaks.
