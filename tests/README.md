# Testing Instructions for PR1

## Setup
1. Install pytest: `pip install pytest pytest-cov`
2. Run from repository root: `pytest tests/test_security_and_types.py -v`

## Expected Results
All tests should pass. Key verifications:

1. **Security Tests**:
   - All exploit attempts return None
   - No actual code execution occurs
   - Safe math formulas still work

2. **Type Safety Tests**:
   - isinstance() calls don't crash
   - Type validation works for all cases

3. **Integration Tests**:
   - Real-world formulas calculate correctly
   - System remains secure

## Coverage
Run with coverage: `pytest tests/test_security_and_types.py --cov=energy_transformer.spec --cov-report=html`
