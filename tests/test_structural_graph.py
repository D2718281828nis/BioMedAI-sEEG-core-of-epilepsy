import numpy as np
import pytest

from multimodal_approach.dicom_geometry import VolumeGeometry
from multimodal_approach.structural_graph import (
    build_structural_anomaly_graph, plot_structural_anomaly_graph, plot_structural_anomaly_graph_anatomical,
)


def _cluster(hemisphere, xyz, mean_abs_anomaly=1.0, total_mass=10.0, peak_value=5.0, voxel_count=20):
    return {
        "voxel_count": voxel_count,
        "total_mass": total_mass,
        "mean_abs_anomaly": mean_abs_anomaly,
        "peak_value": peak_value,
        "peak_voxel_kij": (0, 0, 0),
        "centroid_voxel_kij": (0.0, 0.0, 0.0),
        "peak_patient_xyz_mm": xyz,
        "hemisphere": hemisphere,
    }


def test_build_structural_anomaly_graph_raises_on_no_clusters():
    with pytest.raises(ValueError):
        build_structural_anomaly_graph([], [])


def test_build_structural_anomaly_graph_nodes_and_proximity_edges():
    anomaly_clusters = [
        _cluster("right", (10.0, 0.0, 0.0), mean_abs_anomaly=5.0),
        _cluster("left", (-100.0, 0.0, 0.0), mean_abs_anomaly=2.0),
    ]
    heterogeneity_clusters = [_cluster("right", (12.0, 0.0, 0.0), mean_abs_anomaly=3.0)]

    graph = build_structural_anomaly_graph(anomaly_clusters, heterogeneity_clusters,
                                           distance_threshold_mm=40.0, top_k_per_node=3)

    assert set(graph.nodes) == {"asym_0", "asym_1", "het_0"}
    assert graph.nodes["asym_0"]["channel"] == "asymmetry"
    assert graph.nodes["het_0"]["channel"] == "heterogeneity"
    assert graph.nodes["asym_1"]["hemisphere"] == "left"
    assert graph.nodes["asym_0"]["strength"] == pytest.approx(5.0)

    # asym_0 and het_0 are 2 mm apart (within threshold) -> edge present.
    assert graph.has_edge("asym_0", "het_0")
    edge = graph.edges["asym_0", "het_0"]
    assert edge["kind"] == "proximity"
    assert edge["distance_mm"] == pytest.approx(2.0)
    assert edge["weight"] == pytest.approx(1.0 / 3.0)

    # asym_1 is 110/112 mm away from the others -> beyond distance_threshold_mm, no edge.
    assert not graph.has_edge("asym_0", "asym_1")
    assert not graph.has_edge("asym_1", "het_0")
    assert graph.degree["asym_1"] == 0


def test_build_structural_anomaly_graph_top_k_prunes_extra_neighbours():
    # Four clusters in a tight line 1 mm apart; top_k_per_node=1 should keep
    # each node to its single nearest neighbour, not connect everything.
    anomaly_clusters = [_cluster("right", (float(i), 0.0, 0.0)) for i in range(4)]

    graph = build_structural_anomaly_graph(anomaly_clusters, [], distance_threshold_mm=10.0,
                                           top_k_per_node=1)

    assert graph.number_of_nodes() == 4
    assert all(graph.degree[node] <= 2 for node in graph.nodes)
    assert graph.number_of_edges() < 6  # fewer than the fully-connected 4-choose-2


def test_plot_structural_anomaly_graph_writes_file(tmp_path):
    anomaly_clusters = [_cluster("right", (0.0, 0.0, 0.0)), _cluster("left", (5.0, 0.0, 0.0))]
    graph = build_structural_anomaly_graph(anomaly_clusters, [])

    output_path = tmp_path / "structural_anomaly_graph.png"
    plot_structural_anomaly_graph(graph, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


class _StubResult:
    def __init__(self, t1_geometry, brain_mask):
        self.t1_geometry = t1_geometry
        self.brain_mask = brain_mask


def _synthetic_geometry(shape=(6, 8, 8)):
    # Axis-aligned 2 mm grid: k -> z, i -> y, j -> x (order doesn't matter for
    # this test, only that voxel_to_patient/patient_to_voxel round-trip).
    rng = np.random.default_rng(1)
    volume = rng.uniform(0, 100, size=shape).astype(np.float32)
    return VolumeGeometry(
        volume=volume, origin=np.zeros(3), d_slice=np.array([0.0, 0.0, 2.0]),
        d_row=np.array([0.0, 2.0, 0.0]), d_col=np.array([2.0, 0.0, 0.0]), series=None,
    )


def test_plot_structural_anomaly_graph_anatomical_writes_file(tmp_path):
    geometry = _synthetic_geometry()
    brain_mask = np.ones(geometry.volume.shape, dtype=bool)
    result = _StubResult(geometry, brain_mask)

    # Two clusters at real, round-trippable patient coordinates (derived from
    # actual voxel positions via voxel_to_patient, not made up numbers), so
    # plot_structural_anomaly_graph_anatomical's patient_to_voxel projection
    # lands inside the volume instead of off-canvas.
    xyz_a = [float(c) for c in geometry.voxel_to_patient(2, 3, 3)]
    xyz_b = [float(c) for c in geometry.voxel_to_patient(2, 4, 4)]
    anomaly_clusters = [_cluster("right", xyz_a), _cluster("left", xyz_b)]
    graph = build_structural_anomaly_graph(anomaly_clusters, [], distance_threshold_mm=10.0)
    assert graph.number_of_edges() == 1  # the two synthetic clusters are close by construction

    output_path = tmp_path / "structural_anomaly_graph_anatomical.png"
    plot_structural_anomaly_graph_anatomical(result, graph, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
