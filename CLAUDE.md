# sppricer: agent brief

Monte Carlo pricer for structured products (numpy only). Before any change, read:

- [CONTRIBUTING.md](CONTRIBUTING.md): conventions, units, testing standard, definition of done
- [ROADMAP.md](ROADMAP.md): what to build next and how each milestone is accepted

## Commands

```bash
pip install -e ".[dev]"
pytest -q                                  # must pass and stay under 30 s
python examples/price_worst_of_phoenix.py  # regenerate README numbers if prices move
```

If the package isn't installed, `PYTHONPATH=src pytest -q` works.

## Rules that are easy to break

- Never loosen a test tolerance or add a fixed allowance to make it pass. Find the cause.
- New options default to the old behaviour; existing prices must not move silently.
- No new runtime dependency without a written reason.
- Mark approximations with a `ponytail:` comment and add them to README "Limitations".
- Work only on the milestone in hand; update its checkboxes in ROADMAP.md when done.
