"""Tests for mister package."""


class TestMister():
    """Test Mister functions."""

    def test_radius(self):
        """Test functions."""
        from mister import Mister

        mr = Mister()
        out = mr.radius([0, 0.1, 0.5])
        assert len(out) == 1
        assert isinstance(out[0], float)

    def test_radius2(self):
        """Test radius with multiple arguments."""
        from mister import Mister

        mr = Mister()
        out = mr.radius([[0.5, 1.0, 0.75], [-3, 10.0, 0.25]])
        assert len(out) == 2
        assert isinstance(out[0], float)

    def test_lifetime(self):
        """Test lifetime."""
        from mister import Mister

        mr = Mister()
        out = mr.lifetime([0, 0.3])
        assert len(out) == 1
        assert isinstance(out[0], float)
