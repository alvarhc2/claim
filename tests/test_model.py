"""Shapes, masks and the published parameter counts."""

from __future__ import annotations

import math
import unittest

import torch

from claim.block import LOCALITY_SCALE, ClaimBlock
from claim.models import build
from claim.published import FAMILY_FEATURES, columns


class BlockTests(unittest.TestCase):
    def test_shape_and_gradient(self) -> None:
        block = ClaimBlock(12, 6, 4, variant="full", locality="exact")
        sequence = torch.randn(3, 12, 6)
        output = block(sequence)
        self.assertEqual(output.shape, sequence.shape)
        output.sum().backward()
        self.assertGreater(float(block.alpha.grad.abs()), 0.0)
        self.assertGreater(float(block.gamma.grad.abs()), 0.0)

    def test_wide_gaussian_keeps_distant_tokens(self) -> None:
        length = 32
        block = ClaimBlock(length, 4, length, variant="full", locality="exact")
        with torch.no_grad():
            block.alpha.fill_(20.0)
            mask = block.locality_mask()
        distance = (length - 1) ** 2
        spread = LOCALITY_SCALE * length
        expected = math.exp(-distance / (2.0 * spread**2))
        self.assertAlmostEqual(float(mask[length - 1, 0]), expected, places=4)

    def test_lookup_matches_exact_at_the_ends(self) -> None:
        exact = ClaimBlock(20, 4, 6, variant="full", locality="exact")
        lookup = ClaimBlock(20, 4, 6, variant="full", locality="lookup", lookup_bins=10)
        with torch.no_grad():
            for value in (-20.0, 20.0):
                exact.alpha.fill_(value)
                lookup.alpha.fill_(value)
                self.assertTrue(torch.allclose(exact.locality_mask(), lookup.locality_mask(), atol=1e-5))

    def test_causality_blocks_the_future(self) -> None:
        block = ClaimBlock(10, 4, 4, variant="full")
        with torch.no_grad():
            block.beta.fill_(20.0)
            mask = block.causality_mask()
        future = mask[~block.past]
        self.assertTrue(torch.all(future < 1e-5))

    def test_shared_kernel_when_gamma_is_closed(self) -> None:
        block = ClaimBlock(9, 4, 5, variant="location-invariance")
        with torch.no_grad():
            block.gamma.fill_(-20.0)
            projection = block.projection()
        for row in range(projection.shape[0]):
            shifted = block.shared.detach()[block.shift[row]]
            self.assertTrue(torch.allclose(projection[row], shifted, atol=1e-5))

    def test_folded_forward_matches(self) -> None:
        torch.manual_seed(0)
        block = ClaimBlock(8, 4, 3, variant="full", locality="exact")
        sequence = torch.randn(2, 8, 4)
        block.eval()
        eager = block(sequence)
        block.fold_inference()
        folded = block(sequence)
        self.assertTrue(torch.allclose(eager, folded, atol=1e-5))


class CountTests(unittest.TestCase):
    def test_published_counts(self) -> None:
        expected = {
            ("ecg200", "claim", 1): 41670,
            ("ecg200", "mixer", 1): 82834,
            ("ecg200", "transformer", 1): 162717,
            ("kepler", "claim", 1): 641282,
            ("kepler", "mixer", 1): 1268528,
            ("kepler", "transformer", 1): 313778,
            ("tess", "claim", 1): 27356,
            ("tess", "mixer", 1): 48906,
            ("tess", "transformer", 1): 30948,
            ("icu", "claim", FAMILY_FEATURES["amg"]): 3524,
        }
        for (dataset, model, features), count in expected.items():
            specs, embed, scalar = columns(dataset, model)
            network = build(model, features, specs, embed, scalar)
            got = sum(parameter.numel() for parameter in network.parameters())
            self.assertEqual(got, count, f"{dataset} {model}")


if __name__ == "__main__":
    unittest.main()
