-- MFE-based 2h scratch rule (P3 fix 2026-04-22).
--
-- Course Lesson 13 [47:00]:
--   "If you're not in substantial profit within two hours you scratch the
--    trade. It means the Market Maker has a different plan."
--
-- "Within two hours" is a window, not an instant. The previous rule
-- measured unrealized P&L at the 2h mark, which closed trades that had
-- been +1R mid-flight but pulled back to break-even at the check —
-- violating the course intent.
--
-- This column records the highest R-multiple the trade ever reached.
-- If mm_max_favorable_excursion_r >= threshold (default 0.3R) by the
-- 2h mark, the trade has shown "substantial profit" during the window
-- and is safe from the scratch rule, even if it's currently flat.
--
-- Persisted so a restart mid-trade doesn't lose the fact that the
-- trade already cleared the bar.

ALTER TABLE trades ADD COLUMN IF NOT EXISTS mm_max_favorable_excursion_r numeric DEFAULT 0;
