from __future__ import annotations

import os

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def _base_url() -> str:
    value = os.getenv("RESEARCH_DESK_BASE_URL")
    if not value:
        pytest.skip("RESEARCH_DESK_BASE_URL must point to a running workbench")
    return value.rstrip("/")


def _complete_demo(page: Page, *, collapse_sidebar: bool = False) -> None:
    page.goto(_base_url(), wait_until="networkidle")
    expect(page).to_have_title("Research Desk")
    expect(page.get_by_role("heading", name="Research Desk", exact=True, level=1)).to_be_visible()
    expect(page.get_by_text("Demo mode", exact=False).first).to_be_visible()
    sidebar = page.get_by_test_id("stSidebar")
    if collapse_sidebar and sidebar.get_attribute("aria-expanded") == "true":
        page.get_by_test_id("stSidebarCollapseButton").get_by_role("button").click()
        expect(sidebar).to_have_attribute("aria-expanded", "false")
    page.get_by_label("Research topic").fill(
        "How should multi-agent research systems be evaluated?"
    )
    page.get_by_role("button", name="Start research").click()
    expect(page.get_by_text("Completed research", exact=False)).to_be_visible(timeout=30_000)
    expect(page.get_by_role("heading", name="Full report")).to_be_visible()
    expect(page.get_by_text("[S1]", exact=False).first).to_be_visible()


def test_complete_research_journey_has_no_browser_errors(page: Page) -> None:
    errors: list[str] = []

    def record_console_error(message: object) -> None:
        if getattr(message, "type", "") == "error":
            errors.append(str(getattr(message, "text", message)))

    page.on("console", record_console_error)
    page.on("pageerror", lambda error: errors.append(str(error)))

    _complete_demo(page)
    page.get_by_role("tab", name="Sources").click()
    expect(page.get_by_role("heading", name="Source library")).to_be_visible()
    expect(page.get_by_role("button", name="Markdown")).to_be_visible()
    assert errors == []


def test_mobile_layout_does_not_overflow(page: Page) -> None:
    page.set_viewport_size({"width": 390, "height": 844})
    _complete_demo(page, collapse_sidebar=True)
    dimensions = page.evaluate(
        "() => ({ width: document.documentElement.scrollWidth, viewport: window.innerWidth })"
    )
    assert dimensions["width"] <= dimensions["viewport"] + 1
