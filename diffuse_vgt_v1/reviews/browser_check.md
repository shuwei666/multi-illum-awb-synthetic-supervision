# Local report browser check

updated: 2026-09-26; agent: codex (main executor, formatting self-check)

Actual Chrome opened the local report through a temporary loopback HTTP server. The automation browser blocks file URLs; this was not treated as a report failure. Title: `Diffuse Virtual GT v1：seed0结果与证据边界`. No external publication occurred.

| Requested viewport | Actual content viewport (clientWidth) | document scrollWidth | body scrollWidth |
| --- | --- | --- | --- |
| 1400 | 1385 | 1385 | 1385 |
| 768 | 753 | 753 | 753 |
| 375 | 360 | 360 | 360 |

The 15-pixel difference is the visible native vertical scrollbar, not horizontal overflow. Both document and body match the usable content viewport. Horizontal overflow is visible, not hidden or clipped. At 768/375 the tables become labeled cards. Main executor viewed the actual top/TOC and table screenshots at these widths. Full-precision table numbers wrap on desktop; no numbers are clipped. The sole console resource error is the absent optional favicon.

Screenshots: `output/playwright/report-1400.png`, `report-768.png`, `report-375.png`, and corresponding `report-1400-table.png`, `report-768-table.png`, `report-375-table.png` in that directory. This formatting check does not supersede numerical review, local content acceptance, or the public-export hold.

HTML SHA256: `74afca7c4c432f596a37da6658c07aa21b09e0d0e5131b32f1379a3353877c31`.
