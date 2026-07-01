"""Lythéa tools — outils externes invocables par le routeur V5.1.

Each tool is a self-contained module with a clear interface :
- input : dict of arguments (string-keyed)
- output : dict containing at least ``ok: bool`` and either
  ``result`` (success) or ``error`` (failure), plus optional
  metadata (stdout, stderr, duration, files produced, …).

V5.1 ships with :
- python_executor : run Python code in a sandboxed subprocess
- (web is still handled by lythea.web — historical, will be
  migrated to a tool here in V5.2 for uniformity)

Adding a new tool : create a module here, expose a ``run(args)``
function, register a Route in semantic_router.ROUTES and a tool
name in tool_dispatcher.VALID_TOOLS.
"""
