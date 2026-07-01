"""Tests Rune — suite sans GPU (MockBackend).

Couvre :
- Backend Mock (encode, generate, hooks)
- WorkingMemoryBuffer (capacité, éviction, fraîcheur)
- TieredRetriever (fallback strict, doubt gate)
- AutoSkillStore (add, dedup, find, record_failure)
- FailureMemory (add, find, warning_block)
- SurpriseMeter (compute_input, compute_output)
- Metacognition (observe, calibration)
- CognitiveLoop (process, status)
- SubAgentSpawner (run, timeout)
- CronScheduler (add, schedule, execute)
- API (health, chat, status)
- CLI (smoke tests via CliRunner)
"""
