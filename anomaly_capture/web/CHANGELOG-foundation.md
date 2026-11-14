# Foundation changelog

Changes to shared classes, tokens or the core API after `.split_done` (newest last).

- 04:23 split done. Single near-black theme (no light theme, no toggle). Live images start after window load (needed for headless screenshots).
- 04:27 narrow layout (<900px): sidebar scroller is static with a hidden scrollbar (was sticky; its scrollbar showed as a light strip); top bar sticks at top:0. Hash deep links /#/later/sort route client-side.
- 04:33 overview rows use state classes `.jrow.is-done|is-part|is-todo|is-rec|is-run|is-later` (plain `todo` collided with the `.todo` "What to do" block). Rule: never use bare state words (todo, done, next...) as classes on elements outside your page; prefix them.
- 04:36 `.seg button` no longer wraps; `.seg.full button` padding 0 8px.
- 04:40 DESIGN.md: URLs/restarts section, rule 8 (global class names), later-page template.
- 04:42 foundation done. /later/* and /stills need ONE server restart to load directly (arm was connected, so no restart was done); until then use in-app navigation or /#/later/sort.
