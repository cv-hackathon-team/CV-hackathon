"""Streamlit demo: compare depth upsampling methods on a selected sample."""

from __future__ import annotations

from typing import Any


def load_model(checkpoint_path: str) -> Any:
    """Load a DepthGuardNet checkpoint, cached across reruns."""
    # TODO: implement with st.cache_resource.
    raise NotImplementedError


def render_sidebar() -> dict[str, Any]:
    """Render the sample, scale, corruption and method controls."""
    # TODO: implement.
    raise NotImplementedError


def render_results(results: dict[str, Any]) -> None:
    """Show depth maps, confidence, boundaries, diagnosis and metrics."""
    # TODO: implement.
    raise NotImplementedError


def main() -> None:
    """Run the app."""
    # TODO: implement.
    raise NotImplementedError


if __name__ == "__main__":
    main()
