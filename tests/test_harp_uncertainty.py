"""Regional variance aggregation and its optional plot labels."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import foxes


def box(harpnum=1234, *, top=0, left=0, width=100, height=100):
    return {"HARPNUM": harpnum, "top_pct": top, "left_pct": left,
            "width_pct": width, "height_pct": height}


def prediction(regions, *, with_uncertainty=True, metadata=None):
    return foxes.Prediction(
        metadata=metadata, flux=1e-7,
        flux_map=np.array([[1, 2], [3, 4]]) * 1e-8,
        patch_uncertainty_map=(np.array([[3, 4], [0, 12]]) * 1e-9 if with_uncertainty else None),
        harp_regions=regions,
    )


def test_harp_quadrature_matches_flux_geometry_and_does_not_mutate_regions():
    regions = [box(), box(2, top=50, height=50), box(3, height=50), box(4, width=10, height=10)]
    result = prediction(regions)
    assert [row["flux_sum"] for row in result.harp_regions] == pytest.approx([1e-7, 3e-8, 7e-8, 0])
    # sqrt(3**2 + 4**2 + 12**2) = 13, not the sum of sigmas (19).
    assert [row["flux_sum_uncertainty"] for row in result.harp_regions] == pytest.approx(
        [13e-9, 5e-9, 12e-9, 0], abs=1e-20,
    )
    assert all("flux_sum_uncertainty" not in row for row in regions)


def test_variances_are_squared_in_float64_to_preserve_small_values():
    result = foxes.Prediction(
        flux=1e-20, flux_map=np.ones((1, 2)) * 5e-21,
        patch_uncertainty_map=np.array([[3, 4]], dtype=np.float32) * 1e-30,
        harp_regions=[box()],
    )
    assert result.harp_regions[0]["flux_sum_uncertainty"] == pytest.approx(5e-30, rel=1e-6, abs=0)


def test_deterministic_prediction_has_no_uncertainty_instead_of_reusing_stale_value():
    region = {**box(), "flux_sum_uncertainty": 9e-9}
    result = prediction([region], with_uncertainty=False)
    assert result.harp_regions[0]["flux_sum_uncertainty"] is None
    assert region["flux_sum_uncertainty"] == 9e-9


@pytest.mark.parametrize("uncertainty,match", [
    (np.ones((3, 3)), "match flux_map shape"),
    (np.array([[0, 0], [0, np.nan]]), "finite and nonnegative"),
    (np.array([[0, 0], [0, -1]]), "finite and nonnegative"),
])
def test_prediction_rejects_invalid_uncertainty_maps(uncertainty, match):
    with pytest.raises(ValueError, match=match):
        foxes.Prediction(flux=4e-9, flux_map=np.ones((2, 2)) * 1e-9,
                         patch_uncertainty_map=uncertainty, harp_regions=[box()])


@pytest.mark.parametrize("show_uncertainty", [False, True])
@pytest.mark.parametrize("with_uncertainty", [False, True])
def test_uncertainty_label_toggle_does_not_change_results(show_uncertainty, with_uncertainty):
    result = prediction([box()], with_uncertainty=with_uncertainty)
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0,
                             show_harp_uncertainty=show_uncertainty)
    text = "#1234\n1.00e-07"
    if show_uncertainty and with_uncertainty:
        text += "\n± 1.30e-08"
    assert ax.texts[0].get_text() == text
    assert result.harp_regions[0]["flux_sum_uncertainty"] == (pytest.approx(13e-9) if with_uncertainty else None)
    plt.close(ax.figure)


@pytest.mark.parametrize("deferred", [False, True])
def test_explicit_and_deferred_boxes_get_uncertainty_from_prediction_map(monkeypatch, deferred):
    metadata = foxes.ObservationMetadata(timestamp="2023-08-01", image_shape=(7, 2, 2))
    result = prediction(None, metadata=metadata)
    calls = []

    def fetch(self, timestamp):
        calls.append(timestamp)
        return [box(top=50, height=50)]

    monkeypatch.setattr(foxes.HARPProcessor, "fetch_date", fetch)
    kwargs = {} if deferred else {"harp_regions": [box(top=50, height=50)]}
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0, **kwargs)
    assert ax.texts[0].get_text() == "#1234\n3.00e-08\n± 5.00e-09"
    assert calls == ([metadata.timestamp] if deferred else [])
    assert result.harp_regions is None
    plt.close(ax.figure)


def test_manual_offsets_keep_uncertainty_labels_readable_after_resize():
    regions = [box(10000 + i, left=48 + i * 2, top=48, width=1, height=1)
               for i in range(3)]
    result = foxes.Prediction(flux=4.096e-6, flux_map=np.full((64, 64), 1e-9),
                              patch_uncertainty_map=np.full((64, 64), 2e-10), harp_regions=regions)
    offsets = {10000: (-100, 30), 10001: (0, -70), 10002: (100, 30)}
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0,
                             harp_label_offsets=offsets, harp_label_fontsize=10)
    for size, dpi in [((7, 6), 100), ((5.5, 5.5), 150)]:
        ax.figure.set_size_inches(*size)
        ax.figure.set_dpi(dpi)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        bounds = ax.get_window_extent(renderer)
        boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in ax.texts]
        for i, (label, extent) in enumerate(zip(ax.texts, boxes)):
            assert label.get_fontsize() == 10
            assert "± " in label.get_text()
            assert label.get_position() == offsets[int(label.get_text().splitlines()[0].lstrip("#"))]
            assert extent.x0 >= bounds.x0 and extent.x1 <= bounds.x1
            assert extent.y0 >= bounds.y0 and extent.y1 <= bounds.y1
            assert all(not extent.overlaps(other) for other in boxes[:i])
    plt.close(ax.figure)


@pytest.mark.parametrize("top,left", [(45, 45), (0, 0), (90, 90)])
@pytest.mark.parametrize("with_uncertainty", [False, True])
def test_colliding_labels_shift_without_overlap_or_font_shrinking(top, left, with_uncertainty):
    regions = [box(10000 + i, top=top, left=left, width=3, height=3) for i in range(4)]
    result = foxes.Prediction(
        flux=4.096e-6, flux_map=np.full((64, 64), 1e-9), harp_regions=regions,
        patch_uncertainty_map=np.full((64, 64), 2e-10) if with_uncertainty else None,
    )
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0, harp_label_fontsize=10)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    bounds = [label.get_bbox_patch().get_window_extent(renderer) for label in ax.texts]
    assert len(bounds) == 4
    assert not ax.texts[0].arrow_patch.get_visible()
    for i, label in enumerate(ax.texts):
        assert label.get_visible()
        assert label.get_fontsize() == 10
        assert ("±" in label.get_text()) is with_uncertainty
        assert all(not bounds[i].overlaps(other) for other in bounds[:i])
        if i:
            assert label.arrow_patch.get_visible()
    plt.close(ax.figure)


def test_small_label_overlap_gets_a_small_nudge_instead_of_a_full_height_jump():
    result = prediction([box(10000, top=30, left=20, width=5, height=5),
                         box(10001, top=30, left=20, width=5, height=5)])
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0, harp_label_max_shift=0)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    height = ax.texts[0].get_bbox_patch().get_window_extent(renderer).height / renderer.points_to_pixels(1)
    plt.close(ax.figure)

    # Start the second label just one point inside the first label's bottom edge.
    start = (3, -3 - height + 1)
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0,
                             harp_label_offsets={10001: start})
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    first, second = [label.get_bbox_patch().get_window_extent(renderer) for label in ax.texts]
    assert not first.overlaps(second)
    displacement = np.linalg.norm(np.asarray(ax.texts[1].get_position()) - start)
    assert 0 < displacement <= 8
    assert displacement < height / 2
    plt.close(ax.figure)


@pytest.mark.parametrize("max_shift", [0, 8, 24])
def test_automatic_label_movement_stays_within_limit(max_shift):
    regions = [box(10000 + i, top=45, left=45, width=3, height=3) for i in range(6)]
    result = prediction(regions)
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0,
                             harp_label_max_shift=max_shift)
    for label in ax.texts:
        corner_offset = (-3 if label.get_ha() == "right" else 3,
                         3 if label.get_va() == "bottom" else -3)
        assert np.linalg.norm(np.asarray(label.get_position()) - corner_offset) <= max_shift
        assert label.get_visible()
    plt.close(ax.figure)


@pytest.mark.parametrize("max_shift", [-1, np.inf, np.nan])
def test_invalid_label_shift_limit_is_rejected(max_shift):
    with pytest.raises(ValueError, match="harp_label_max_shift must be finite and nonnegative"):
        foxes.plot_flux_map(np.ones((2, 2)), harp_label_max_shift=max_shift)


@pytest.mark.parametrize("top,left", [(10, 10), (10, 75), (75, 10), (75, 75)])
def test_default_label_is_outside_its_box_at_a_corner(top, left):
    ax = foxes.plot_flux_map(prediction([box(top=top, left=left, width=10, height=10)]),
                             show_harp_boxes=True, min_harp_flux=0)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    label = ax.texts[0]
    region = ax.patches[0].get_bbox()
    assert label.xy[0] in (region.x0, region.x1)
    assert label.xy[1] in (region.y0, region.y1)
    text_bounds = label.get_bbox_patch().get_window_extent(renderer)
    box_bounds = ax.patches[0].get_window_extent(renderer)
    assert text_bounds.x1 < box_bounds.x0 or text_bounds.x0 > box_bounds.x1
    assert text_bounds.y1 < box_bounds.y0 or text_bounds.y0 > box_bounds.y1
    plt.close(ax.figure)


def test_other_corners_are_tried_before_nudges():
    ax = foxes.plot_flux_map(prediction([box(10000, top=40, left=40, width=20, height=20),
                                       box(10001, top=40, left=40, width=20, height=20)]),
                             show_harp_boxes=True, min_harp_flux=0, harp_label_fontsize=10)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    first, second = ax.texts
    assert first.xy != second.xy
    assert all(abs(value) == 3 for value in second.get_position())
    assert not first.get_bbox_patch().get_window_extent(renderer).overlaps(
        second.get_bbox_patch().get_window_extent(renderer),
    )
    plt.close(ax.figure)


def test_small_harp_label_avoids_a_larger_preceding_box():
    result = prediction([box(10000, top=40, left=40, width=20, height=20),
                         box(10001, top=39, left=61, width=2, height=2)])
    ax = foxes.plot_flux_map(result, show_harp_boxes=True, min_harp_flux=0)
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for label in ax.texts:
        bounds = label.get_bbox_patch().get_window_extent(renderer)
        assert all(not bounds.overlaps(region.get_window_extent(renderer)) for region in ax.patches)
    plt.close(ax.figure)
