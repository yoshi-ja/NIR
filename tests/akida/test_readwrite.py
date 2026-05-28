import numpy as np

import nir
from nir.akida import AkidaTargetProfile, AkidaValidator, QuantizationMetadata


def test_akida_quantization_metadata_survives_roundtrip(tmp_path):
    quant_metadata = QuantizationMetadata(
        weight_bits=8,
        activation_bits=4,
        input_bits=8,
        per_channel=True,
    )

    linear = nir.Linear(weight=np.eye(4))
    linear.metadata.update(quant_metadata.to_dict())

    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type=np.array([4])),
            "linear": linear,
            "lif": nir.LIF(
                tau=np.ones(4),
                r=np.ones(4),
                v_leak=np.zeros(4),
                v_threshold=np.ones(4),
                v_reset=np.zeros(4),
            ),
            "output": nir.Output(output_type=np.array([4])),
        },
        edges=[("input", "linear"), ("linear", "lif"), ("lif", "output")],
        type_check=False,
    )

    path = tmp_path / "akida_roundtrip.nir"
    nir.write(path, graph)

    loaded = nir.read(path, type_check=False)

    assert QuantizationMetadata.from_node(loaded.nodes["linear"]) == quant_metadata
    assert loaded.nodes["linear"].metadata["_akida_qm_version"] == "1"
    assert AkidaValidator(AkidaTargetProfile.v1()).validate(loaded).is_valid
