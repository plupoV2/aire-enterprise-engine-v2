# v5 Notes

### Key enterprise features
- workspace_settings: folders + scoring profile + webhook URL
- deal_notes: workflow collaboration (notes/tags/assignee)
- webhook events: deal_saved, deal_moved, deal_re_evaluated, deal_note_added, role_updated, etc.

### IRR stability fix
v4/v3 style Newton-based IRR can overflow on long monthly series if the solver jumps to extreme rates.
v5 uses a bracketing grid + bisection and safe NPV discounting:
- discount = exp(t * log1p(r)) with overflow guards