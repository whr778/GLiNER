"""Tests for the bipartite (trigger, argument) adjacency layer used by events."""

import torch
import pytest

from gliner.modeling.multitask.relations_layers import (
    bipartite_dot_product_adjacency,
    BipartiteMLPDecoder,
    BipartiteBilinearDecoder,
    BipartiteRelationsRepLayer,
)


class TestBipartiteDotProductAdjacency:
    def test_output_shape(self):
        B, T, A, D = 2, 3, 4, 8
        trigger_rep = torch.randn(B, T, D)
        arg_rep = torch.randn(B, A, D)

        adj = bipartite_dot_product_adjacency(trigger_rep, arg_rep)

        assert adj.shape == (B, T, A)

    def test_values_in_unit_interval(self):
        trigger_rep = torch.randn(2, 3, 8) * 5
        arg_rep = torch.randn(2, 4, 8) * 5

        adj = bipartite_dot_product_adjacency(trigger_rep, arg_rep)

        assert torch.all(adj >= 0.0)
        assert torch.all(adj <= 1.0)

    def test_mask_zeroes_invalid_entries(self):
        B, T, A, D = 1, 2, 2, 4
        trigger_rep = torch.randn(B, T, D)
        arg_rep = torch.randn(B, A, D)
        trigger_mask = torch.tensor([[1, 0]])
        arg_mask = torch.tensor([[1, 1]])

        adj = bipartite_dot_product_adjacency(trigger_rep, arg_rep, trigger_mask, arg_mask)

        assert torch.all(adj[0, 1, :] == 0.0)
        assert torch.any(adj[0, 0, :] != 0.0)

    def test_no_mask_is_identity_on_masking(self):
        trigger_rep = torch.randn(1, 2, 4)
        arg_rep = torch.randn(1, 3, 4)

        adj_no_mask = bipartite_dot_product_adjacency(trigger_rep, arg_rep)
        adj_full_mask = bipartite_dot_product_adjacency(
            trigger_rep, arg_rep, torch.ones(1, 2), torch.ones(1, 3)
        )

        assert torch.allclose(adj_no_mask, adj_full_mask)

    def test_normalize_option_changes_scores(self):
        trigger_rep = torch.randn(1, 2, 4) * 10
        arg_rep = torch.randn(1, 3, 4) * 10

        adj_raw = bipartite_dot_product_adjacency(trigger_rep, arg_rep, normalize=False)
        adj_norm = bipartite_dot_product_adjacency(trigger_rep, arg_rep, normalize=True)

        assert not torch.allclose(adj_raw, adj_norm)


class TestBipartiteMLPDecoder:
    def test_output_shape_and_range(self):
        B, T, A, D, H = 2, 3, 5, 8, 16
        decoder = BipartiteMLPDecoder(D, H)
        trigger_rep = torch.randn(B, T, D)
        arg_rep = torch.randn(B, A, D)

        adj = decoder(trigger_rep, arg_rep)

        assert adj.shape == (B, T, A)
        assert torch.all(adj >= 0.0) and torch.all(adj <= 1.0)

    def test_mask_zeroes_invalid_entries(self):
        decoder = BipartiteMLPDecoder(4, 8)
        trigger_rep = torch.randn(1, 2, 4)
        arg_rep = torch.randn(1, 2, 4)
        trigger_mask = torch.tensor([[1, 0]])

        adj = decoder(trigger_rep, arg_rep, trigger_mask=trigger_mask)

        assert torch.all(adj[0, 1, :] == 0.0)


class TestBipartiteBilinearDecoder:
    def test_output_shape_and_range(self):
        B, T, A, D, L = 2, 3, 5, 8, 16
        decoder = BipartiteBilinearDecoder(D, L)
        trigger_rep = torch.randn(B, T, D)
        arg_rep = torch.randn(B, A, D)

        adj = decoder(trigger_rep, arg_rep)

        assert adj.shape == (B, T, A)
        assert torch.all(adj >= 0.0) and torch.all(adj <= 1.0)

    def test_mask_zeroes_invalid_entries(self):
        decoder = BipartiteBilinearDecoder(4, 8)
        trigger_rep = torch.randn(1, 2, 4)
        arg_rep = torch.randn(1, 2, 4)
        arg_mask = torch.tensor([[0, 1]])

        adj = decoder(trigger_rep, arg_rep, arg_mask=arg_mask)

        assert torch.all(adj[0, :, 0] == 0.0)


class TestBipartiteRelationsRepLayer:
    @pytest.mark.parametrize("mode", ["dot", "mlp", "bilinear"])
    def test_supported_modes_produce_correct_shape(self, mode):
        B, T, A, D = 2, 3, 4, 8
        layer = BipartiteRelationsRepLayer(D, mode)
        trigger_rep = torch.randn(B, T, D)
        arg_rep = torch.randn(B, A, D)

        adj = layer(trigger_rep, arg_rep)

        assert adj.shape == (B, T, A)

    def test_mode_is_case_insensitive(self):
        layer = BipartiteRelationsRepLayer(4, "DOT")
        adj = layer(torch.randn(1, 2, 4), torch.randn(1, 3, 4))
        assert adj.shape == (1, 2, 3)

    @pytest.mark.parametrize("mode", ["gcn", "gat", "attention", "unknown"])
    def test_unsupported_modes_raise(self, mode):
        """gcn/gat need message-passing over one homogeneous graph, not a
        bipartite pair of sets, so they are deliberately not offered here."""
        with pytest.raises(ValueError):
            BipartiteRelationsRepLayer(4, mode)

    def test_forward_respects_masks(self):
        layer = BipartiteRelationsRepLayer(4, "dot")
        trigger_rep = torch.randn(1, 2, 4)
        arg_rep = torch.randn(1, 2, 4)
        trigger_mask = torch.tensor([[1, 0]])
        arg_mask = torch.tensor([[1, 1]])

        adj = layer(trigger_rep, arg_rep, trigger_mask=trigger_mask, arg_mask=arg_mask)

        assert torch.all(adj[0, 1, :] == 0.0)
