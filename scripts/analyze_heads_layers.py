"""Independent NumPy calculation of the head maps and cumulative rollout."""
import numpy as np

def spatial(p):
    mass = np.sum(p, axis=-1, keepdims=True)
    return np.divide(p, mass, out=np.zeros_like(p), where=mass > 0)

def entropy(p):
    p = spatial(p)
    return -(p * np.log(np.maximum(p, 1e-300))).sum(axis=-1) / np.log(p.shape[-1])

def analyze(a, patch):
    layers, heads, n, _ = a.shape
    grid = 224 // patch
    fused = a.mean(axis=1)
    residual = fused + np.eye(n)
    residual /= residual.sum(axis=-1, keepdims=True)
    joint = np.eye(n)
    joints, metrics = [], []
    headmaps = a[:, :, 0, 1:]
    for layer in range(layers):
        joint = residual[layer] @ joint
        joints.append(joint.copy())
        hm = headmaps[layer]
        mass = hm.sum(axis=-1)
        h = entropy(hm)
        weights = mass / mass.sum()
        mean_h = float(entropy(fused[layer, 0, 1:]))
        weighted_h = float(weights @ h)
        rows = spatial(joint[:, 1:])
        tv = .5 * np.abs(rows - rows[0]).sum(axis=-1)
        mixture = spatial(fused[layer, 0, 1:])
        assert np.max(np.abs(weights @ spatial(hm) - mixture)) < 1e-12
        metrics.append({'Layer': layer + 1, 'HeadEntropies': h.tolist(),
            'HeadPatchMasses': mass.tolist(), 'SpatialHeadMixtureWeights': weights.tolist(),
            'WeightedHeadEntropy': weighted_h, 'MeanHeadEntropy': mean_h,
            'HeadJensenShannon': mean_h - weighted_h,
            'RolloutEntropy': float(entropy(joint[0, 1:])),
            'RolloutPatchMass': float(joint[0, 1:].sum()),
            'MaxRowTVFromCLS': float(tv.max()), 'MeanRowTVFromCLS': float(tv.mean())})
    previous = joints[-2]
    row_mass = previous[:, 1:].sum(axis=-1)
    conditional_rows = spatial(previous[:, 1:])
    weights = residual[-1, 0] * row_mass
    weights /= weights.sum()
    final_spatial = spatial(joint[0, 1:])
    mixture_error = float(np.max(np.abs(weights @ conditional_rows - final_spatial)))
    assert mixture_error < 1e-12
    selected = np.argsort(-fused[-1, 0, 1:], kind='stable')[:2] + 1
    footprints = [{'Token': int(j + 1), 'PatchRow': int((j - 1) // grid + 1),
        'PatchColumn': int((j - 1) % grid + 1),
        'LastMeanAttention': float(fused[-1, 0, j]), 'FinalSpatialMixtureWeight': float(weights[j]),
        'PreviousRolloutEntropy': float(entropy(previous[j, 1:])),
        'Map': conditional_rows[j].reshape(grid, grid).tolist()} for j in selected]
    return {'LayerCount': layers, 'HeadCount': heads, 'PatchGrid': [grid, grid], 'PatchPixels': patch,
        'HeadPatchMaps': headmaps.reshape(layers, heads, grid, grid).tolist(),
        'MeanPatchMaps': fused[:, 0, 1:].reshape(layers, grid, grid).tolist(),
        'RolloutPatchMaps': np.asarray(joints)[:, 0, 1:].reshape(layers, grid, grid).tolist(),
        'LayerMetrics': metrics, 'PreviousCLSFootprint': conditional_rows[0].reshape(grid, grid).tolist(),
        'SelectedFootprints': footprints,
        'FinalMixture': {'MaximumError': mixture_error, 'WeightSum': float(weights.sum()),
            'MaxPreviousRowTVFromCLS': metrics[-2]['MaxRowTVFromCLS'],
            'FinalTVFromPreviousCLS': float(.5 * np.abs(final_spatial - conditional_rows[0]).sum()),
            'SpatialMixtureWeights': weights.tolist()},
        'Scope': 'All heads and layers. Attention routing estimates; no class-specific attribution.'}
