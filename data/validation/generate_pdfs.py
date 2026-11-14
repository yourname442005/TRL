#!/usr/bin/env python3
"""Regenerate the Milestone 3 PDF validation fixtures (09 and 10).

Deterministic: same inputs -> byte-stable layout (pymupdf embeds creation
metadata, so byte equality across runs is not asserted; content is stable).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUT_DIR = Path(__file__).resolve().parent
HEADER = "MERIDIAN AEROSPACE - QUARTERLY TECHNICAL STATUS REPORT"

PAGES: list[tuple[str, list[str]]] = [
    (
        "1. PROGRAM OVERVIEW",
        [
            (
                "The unmanned weather glider program entered its quarterly review "
                "with the sensor payload at an early integration stage."
            ),
            (
                "Basic principles of low-Reynolds-number soaring flight were observed "
                "and recorded in the scientific basis experiment report SNT-2022-009."
            ),
            (
                "The underlying atmospheric thermal-rise phenomenon was characterized "
                "experimentally, and the data were recorded in the flight science notebook."
            ),
        ],
    ),
    (
        "2. CONCEPT AND PROOF OF CONCEPT",
        [
            (
                "The application concept for autonomous storm-chasing gliders was "
                "formulated in a trade study, and the feasibility experiment results "
                "were recorded."
            ),
            (
                "The analytical model of the glide-performance envelope was "
                "experimentally verified against wind-tunnel data, and the verification "
                "results were recorded."
            ),
            (
                "A laboratory demonstration of the critical function was completed at "
                "component level, and the bench test data were recorded."
            ),
        ],
    ),
    (
        "3. LABORATORY VALIDATION",
        [
            (
                "Component validation of the autopilot breadboard was verified in lab "
                "testing, and the results were recorded."
            ),
            (
                "The integration risk assessment was validated against the "
                "ground-station interface test plan, and the findings were recorded."
            ),
        ],
    ),
    (
        "4. RELEVANT ENVIRONMENT TESTING",
        [
            (
                "The component test was executed in a relevant environment, and the "
                "test results show full functionality."
            ),
            (
                "The representative environment fidelity was justified in testing "
                "against the service profile, and the results were recorded."
            ),
        ],
    ),
    (
        "5. STATUS AND NEXT STEPS",
        [
            ("The program office compiled this status report for the quarterly review board."),
            (
                "A system prototype demonstration is planned for next quarter pending "
                "range allocation."
            ),
            (
                "Closed-loop tracking performance remains within the budgeted error "
                "allowance for the campaign."
            ),
        ],
    ),
]


def build_multipage(path: Path) -> None:
    doc = pymupdf.open()
    for page_index, (heading, paragraphs) in enumerate(PAGES, start=1):
        page = doc.new_page()
        page.insert_text((72, 50), HEADER, fontsize=8)
        page.insert_text((72, 100), heading, fontsize=13)
        body = "\n\n".join(paragraphs)
        remaining = page.insert_textbox(pymupdf.Rect(72, 130, 540, 760), body, fontsize=9)
        if remaining < 0:
            raise RuntimeError(f"page {page_index} body overflowed its textbox")
        page.insert_text((300, 800), str(page_index), fontsize=9)
    doc.save(str(path))
    doc.close()


def build_scanned(path: Path) -> None:
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page()
        page.draw_rect(pymupdf.Rect(40, 40, 555, 800), color=(0.6, 0.6, 0.6), width=0.5)
        page.draw_rect(pymupdf.Rect(60, 60, 535, 200), color=(0.75, 0.75, 0.75), width=0.2)
        page.draw_rect(pymupdf.Rect(60, 220, 535, 500), color=(0.75, 0.75, 0.75), width=0.2)
    doc.save(str(path))
    doc.close()


def main() -> None:
    build_multipage(OUT_DIR / "09_multipage_report.pdf")
    build_scanned(OUT_DIR / "10_scanned_image_only.pdf")
    print("wrote 09_multipage_report.pdf and 10_scanned_image_only.pdf")


if __name__ == "__main__":
    main()
