Added a task-tree scoped thread bridge and blocking portal policy stack
(``anyio.thread_policies``) with four independently overridable dimensions –
cancellation handoff (``force_abandon``/``force_wait``), worker thread limiter
binding (``force_default``/``bypass_acquire``), blocking portal admissions
(``reject_new``/``sync_callable_only``) and worker/external thread token gating
(``require_explicit``/``forbid_foreign_token``)
